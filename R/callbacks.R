LightCallback <- R6::R6Class(
  "LightCallback",
  lock_objects = FALSE,
  public = list(
    initialize = function(ctx) {
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
    light_callback_train_valid,
    light_callback_metrics,
    light_callback_progress
  )
}

light_callback <- function(name, ..., public, active, parent_env = parent.frame()) {
  public <- rlang::list2(...)
  R6::R6Class(
    classname = name,
    inherit = LightCallback,
    public = public,
    parent_env = parent_env,
    lock_objects = FALSE
  )
}

light_callback_progress <- light_callback(
  "progress_callback",
  on_train_begin = function() {
    format <- ":current/:total [:bar] - ETA: :eta"
    metrics <- self$ctx$metrics[["train"]][[self$ctx$epoch]]
    abbrevs <- self$get_abbrevs(metrics)
    abbrevs <- paste0(glue::glue("{abbrevs}: :{tolower(abbrevs)} "), collapse = " - ")
    format <- paste0(c(format, abbrevs), collapse = " - ")
    self$pb <- progress::progress_bar$new(
      format = format,
      total = length(self$ctx$data)
    )
  },
  on_epoch_begin = function() {
    rlang::inform(sprintf(
      "Epoch %d/%d",
      as.integer(self$ctx$epoch),
      as.integer(self$ctx$epochs)
    ))
  },
  on_train_batch_end = function() {
    tokens <- self$get_metrics("train")
    names(tokens) <- tolower(names(tokens))
    self$pb$tick(tokens = tokens)
  },
  on_epoch_end = function() {
    self$inform_metrics("train", "Train")
    self$inform_metrics("valid", "Valid")
  },
  get_abbrevs = function(metrics) {
    sapply(metrics, function(x) x$abbrev %||% class(x))
  },
  get_metrics = function(split) {
    metrics <- self$ctx$metrics[[split]][[self$ctx$epoch]]
    l <- list(
      values = sapply(metrics, function(x) x$format(x$compute()))
    )
    names(l) <- self$get_abbrevs(metrics)
    l
  },
  inform_metrics = function(split, name) {
    metrics <- self$get_metrics(split)
    res <- paste0(glue::glue("{names(metrics)}: {metrics}"), collapse = " - ")
    rlang::inform(glue::glue("{name} metrics: {res}"))
  }
)

light_callback_metrics <- light_callback(
  "metrics_callback",
  on_fit_begin = function() {
   self$ctx$metrics <- list(
     train = list(),
     valid = list()
   )
  },
  on_train_begin = function() {
    self$ctx$metrics$train[[self$ctx$epoch]] <- lapply(
      self$ctx$model$metrics %||% list(),
      function(x) x$new()
    )
  },
  on_train_batch_end = function() {
    lapply(
      self$ctx$metrics$train[[self$ctx$epoch]],
      function(x) x$update(self$ctx$pred, self$ctx$target)
    )
  },
  on_valid_begin = function() {
    self$ctx$metrics$valid[[self$ctx$epoch]] <- lapply(
      self$ctx$model$metrics %||% list(),
      function(x) x$new()
    )
  },
  on_valid_batch_end = function() {
    lapply(
      self$ctx$metrics$valid[[self$ctx$epoch]],
      function(x) x$update(self$ctx$pred, self$ctx$target)
    )
  }
)

light_callback_train_valid <- light_callback(
  on_train_begin = function() {
    ctx$model$train()
    ctx$training <- TRUE
  },
  on_valid_begin = function() {
    ctx$model$eval()
    ctx$training <- FALSE
  }
)


