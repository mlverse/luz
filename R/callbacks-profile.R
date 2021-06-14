#' Profile callback
#'
#' Computes the times for high-level operations in the training loops.
#'
#' @note In general you don't need to use these callback by yourself because it's always
#'   included by default in [fit.luz_module_generator()].
#'
#' @family luz_callbacks
#' @export
luz_callback_profile <- luz_callback(
  "profile_callback",
  initialize = function() {
    self$tics <- list()
  },

  on_fit_begin = function() {
    self$tic("fit")
  },
  on_fit_end = function() {
    ctx$log("profile", "fit", self$toc("fit"))
  },

  on_epoch_begin = function() {
    self$tic("epoch")
  },
  on_epoch_end = function() {
    ctx$log("profile", "epoch", self$toc("epoch"))
  },

  on_train_begin = function() {
    self$tic("train")
  },
  on_train_end = function() {
    ctx$log("profile", "train", self$toc("train"))
  },

  on_train_batch_begin = function() {
    self$tic("train_batch")
  },
  on_train_batch_end = function() {
    ctx$log("profile", "train_batch", self$toc("train_batch"))
  },

  on_valid_begin = function() {
    self$tic("valid")
  },
  on_valid_end = function() {
    ctx$log("profile", "valid", self$toc("valid"))
  },

  on_valid_batch_begin = function() {
    self$tic("valid_batch")
  },
  on_valid_batch_end = function() {
    ctx$log("profile", "valid_batch", self$toc("valid_batch"))
  },

  tic = function(name) {
    if (!is.null(self$tics[[name]]))
      abort("Ticking twice with the same name is not allowed.")
    self$tics[[name]] <- Sys.time()
  },
  toc = function(name) {
    time <- difftime(Sys.time(), self$tics[[name]], units = "secs")
    self$tics[[name]] <- NULL
    as.numeric(time)
  }
)


