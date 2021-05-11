#' @include utils.R

LuzCallback <- R6::R6Class(
  "LuzCallback",
  lock_objects = FALSE,
  public = list(
    initialize = function() {},
    set_ctx = function(ctx) {
      self$ctx <- ctx
    },
    call = function(callback_nm) {
      if (is.null(self[[callback_nm]]))
        return(invisible())
      self[[callback_nm]]()
      invisible()
    }
  )
)

call_all_callbacks <- function(callbacks, name) {
  lapply(callbacks, function(callback) {
    callback$call(name)
  })
}

default_callbacks <- function() {
  list(
    luz_callback_train_valid(),
    luz_callback_metrics(),
    luz_callback_progress()
  )
}

#' Create a new callback
#'
#' @param name name of the callback
#' @param ... Public methods of the callback. The name of the methods is used
#'  to know how they should be called. See the details section.
#' @inheritParams R6::R6Class
#'
#' @includeRmd man/rmd/callbacks.Rmd details
#' @examples
#' print_callback <- luz_callback(
#'  name = "print_callback",
#'  on_train_batch_end = function() {
#'    cat("Iteration ", ctx$iter, "\n")
#'  },
#'  on_epoch_end = function() {
#'    cat("Done!\n")
#'  }
#' )
#' @returns
#' A `luz_callback` that can be passed to [fit.luz_module_generator()].
#' @export
luz_callback <- function(name, ..., private = NULL, active = NULL, parent_env = parent.frame()) {
  public <- rlang::list2(...)
  callback_class <- R6::R6Class(
    classname = name,
    inherit = LuzCallback,
    public = public,
    private = private,
    active = active,
    parent_env = parent_env,
    lock_objects = FALSE
  )
  init <- get_init(callback_class)
  f <- rlang::new_function(
    args = rlang::fn_fmls(init),
    body = rlang::expr({
      obj <- R6::R6Class(
        inherit = callback_class,
        public = list(
          initialize = function() {
            super$initialize(!!!rlang::fn_fmls_syms(init))
          }
        ),
        private = private,
        active = active,
        lock_objects = FALSE
      )
      obj$new()
    })
  )
  attr(f, "r6_class") <- callback_class
  f
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


