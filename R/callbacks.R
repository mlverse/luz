LuzCallback <- R6::R6Class(
  "LuzCallback",
  lock_objects = FALSE,
  public = list(
    set_ctx = function(ctx) {
      self$ctx <- ctx
    },
    call = function(callback_nm) {
      if (is.null(self[[callback_nm]]))
        return(invisible())
      self[[callback_nm]]()
      invisible()
    }
    # on_fit_begin = NULL,
    # on_epoch_begin = NULL,
    # on_train_begin = NULL,
    # on_train_batch_begin = NULL,
    # on_train_batch_after_pred = NULL,
    # on_train_batch_after_loss = NULL,
    # on_train_batch_before_backward = NULL,
    # on_train_batch_before_step = NULL,
    # on_train_batch_after_step = NULL,
    # on_train_batch_end = NULL,
    # on_train_end = NULL,
    # on_valid_begin = NULL,
    # on_valid_batch_begin = NULL,
    # on_valid_batch_after_pred = NULL,
    # on_valid_batch_after_loss = NULL,
    # on_valid_batch_end = NULL,
    # on_valid_end = NULL,
    # on_epoch_end = NULL,
    # on_fit_end = NULL
  )
)

call_all_callbacks <- function(callbacks, name) {
  lapply(callbacks, function(callback) {
    callback$call(name)
  })
}

default_callbacks <- function() {
  list(
    luz_callback_train_valid$new(),
    luz_callback_metrics$new(),
    luz_callback_progress$new()
  )
}

#' Create a new callback
#'
#' @param name nm
#' @param ... public methods
#' @inheritParams R6::R6Class
#'
#' @export
luz_callback <- function(name, ..., private = NULL, active = NULL, parent_env = parent.frame()) {
  public <- rlang::list2(...)
  e <- new.env(parent = parent_env)
  R6::R6Class(
    classname = name,
    inherit = LuzCallback,
    public = public,
    private = private,
    active = active,
    parent_env = e,
    lock_objects = FALSE
  )
}

luz_callback_progress <- luz_callback(
  "progress_callback",
  on_train_begin = function() {
    format <- ":current/:total [:bar] - ETA: :eta"
    metrics <- ctx$metrics[["train"]][[ctx$epoch]]
    if (length(metrics) > 0) {
      abbrevs <- self$get_abbrevs(metrics)
      abbrevs <- paste0(glue::glue("{abbrevs}: :{tolower(abbrevs)} "), collapse = " - ")
    } else {
      abbrevs <- NULL
    }

    format <- paste0(c(format, abbrevs), collapse = " - ")
    self$pb <- progress::progress_bar$new(
      format = format,
      total = length(ctx$data)
    )
  },
  on_epoch_begin = function() {
    inform(sprintf(
      "Epoch %d/%d",
      as.integer(ctx$epoch),
      as.integer(ctx$epochs)
    ))
  },
  on_train_batch_end = function() {
    if (ctx$verbose) {
      tokens <- self$get_metrics("train")
      names(tokens) <- tolower(names(tokens))
      self$pb$tick(tokens = tokens)
    }
  },
  on_epoch_end = function() {
    self$inform_metrics("train", "Train")
    self$inform_metrics("valid", "Valid")
  },
  get_abbrevs = function(metrics) {
    sapply(metrics, function(x) x$abbrev %||% class(x))
  },
  get_metrics = function(split) {

    metrics_split <- ctx$metrics[[split]]
    if (length(metrics_split) >= ctx$epoch) {
      metrics <- ctx$metrics[[split]][[ctx$epoch]]
    } else {
      return(list())
    }

    if (length(metrics) == 0)
      return(list())

    # grab pre-computed values (they might not be available though)
    metric_record <- ctx$records$metrics[[split]]
    if (length(metric_record) >= ctx$epoch) {
      values <- ctx$records$metrics[[split]][[ctx$epoch]]
    } else {
      values <- lapply(metrics, function(x) {
        x$compute()
      })
    }

    # format
    l <- lapply(seq_along(metrics), function(i) {
      metrics[[i]]$format(values[[i]])
    })
    names(l) <- self$get_abbrevs(metrics)
    l
  },
  inform_metrics = function(split, name) {
    metrics <- self$get_metrics(split)
    if (length(metrics) > 0) {
      res <- paste0(glue::glue("{names(metrics)}: {metrics}"), collapse = " - ")
      inform(glue::glue("{name} metrics: {res}"))
    }
  }
)

luz_callback_metrics <- luz_callback(
  "metrics_callback",
  on_fit_begin = function() {
   ctx$metrics <- list(
     train = list(),
     valid = list()
   )
   ctx$losses <- list(
     train = list(),
     valid = list()
   )
   ctx$records$metrics <- list(
     train = list(),
     valid = list()
   )
  },
  on_train_begin = function() {
    ctx$metrics$train[[ctx$epoch]] <- lapply(
      ctx$model$metrics %||% list(),
      self$initialize_metric
    )
  },
  on_train_batch_end = function() {
    lapply(
      ctx$metrics$train[[ctx$epoch]],
      function(x) x$update(ctx$pred, ctx$target)
    )
    ctx$losses$train[[ctx$epoch]] <- ctx$loss
  },
  on_train_end = function() {
    ctx$records$metrics$train[[ctx$epoch]] <- lapply(
      ctx$metrics$train[[ctx$epoch]],
      function(x) x$compute()
    )
  },
  on_valid_begin = function() {
    ctx$metrics$valid[[ctx$epoch]] <- lapply(
      ctx$model$metrics %||% list(),
      self$initialize_metric
    )
  },
  on_valid_batch_end = function() {
    lapply(
      ctx$metrics$valid[[ctx$epoch]],
      function(x) x$update(ctx$pred, ctx$target)
    )
    ctx$losses$valid[[ctx$epoch]] <- ctx$loss
  },
  on_valid_end = function() {
    ctx$records$metrics$valid[[ctx$epoch]] <- lapply(
      ctx$metrics$valid[[ctx$epoch]],
      function(x) x$compute()
    )
  },
  initialize_metric  = function(x) {
    obj <- x$new()
    bind_context(obj, ctx)
    obj
  }
)

luz_callback_train_valid <- luz_callback(
  "train_valid_callback",
  on_train_begin = function() {
    ctx$model$train()
    ctx$training <- TRUE
    ctx$loss <- list()
  },
  on_valid_begin = function() {
    ctx$model$eval()
    ctx$training <- FALSE
    ctx$loss <- list()
  }
)


