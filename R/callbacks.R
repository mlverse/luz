LightCallback <- R6::R6Class(
  "LightCallback",
  lock_objects = FALSE,
  public = list(
    initialize = function(ctx) {
      self$ctx <- ctx
    },
    call = function(callback_nm) {
      if (is.null(private[[callback_nm]]))
        return(invisible())
      private[[callback_nm]]()
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
    light_callback_progress
  )
}

light_callback <- function(name, ..., private, active, parent_env = parent.frame()) {
  private <- rlang::list2(...)
  R6::R6Class(
    classname = name,
    inherit = LightCallback,
    private = private,
    parent_env = parent_env,
    lock_objects = FALSE
  )
}

light_callback_progress <- light_callback(
  "progress_callback",
  on_train_begin = function() {
    self$pb <- progress::progress_bar$new(
      format = ":current/:total [:bar] - ETA: :eta",
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
    self$pb$tick()
  }
)


