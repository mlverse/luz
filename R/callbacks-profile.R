#' Profile callback
#'
#' Computes the times for high-level operations in the training loops.
#'
#' @details
#' Records are saved in `ctx$records$profile`. Times are stored as seconds.
#' Data is stored in the following structure:
#'
#' - **fit** time for the entire fit procedure.
#' - **epoch** times per epoch
#' - **(train/valid)_batch** time per batch of data processed, including data
#'   acquisition and step.
#' - **(train/valid)_step** time per step (training or validation step) - only
#'   the model step. (not including data acquisition and preprocessing)
#'
#' @note In general you don't need to use these callback by yourself because it's always
#'   included by default in [fit.luz_module_generator()].
#'
#' @examples
#' profile_callback <- luz_callback_profile()
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
    self$tic("train_batch")
  },
  on_train_end = function() {
    trash <- self$toc("train_batch") # we don't use the last one, but we still reset it.
    ctx$log("profile", "train", self$toc("train"))
  },

  on_train_batch_begin = function() {
    self$tic("train_step")
  },
  on_train_batch_end = function() {
    ctx$log("profile", "train_step", self$toc("train_step"))
    ctx$log("profile", "train_batch", self$toc("train_batch"))
    self$tic("train_batch")
  },

  on_valid_begin = function() {
    self$tic("valid")
    self$tic("valid_batch")
  },
  on_valid_end = function() {
    trash <- self$toc("valid_batch")
    ctx$log("profile", "valid", self$toc("valid"))
  },

  on_valid_batch_begin = function() {
    self$tic("valid_step")
  },
  on_valid_batch_end = function() {
    ctx$log("profile", "valid_step", self$toc("valid_step"))
    ctx$log("profile", "valid_batch", self$toc("valid_batch"))
    self$tic("valid_batch")
  },

  tic = function(name) {
    if (!is.null(self$tics[[name]]))
      abort("Ticking twice with the same name is not allowed.")
    self$tics[[name]] <- Sys.time()
  },
  toc = function(name) {
    if (is.null(self$tics[[name]])) return()

    time <- difftime(Sys.time(), self$tics[[name]], units = "secs")
    self$tics[[name]] <- NULL
    as.numeric(time)
  }
)

get_total_time <- function(x) {
  unlist(x$ctx$records$profile$fit)
}

get_average_time <- function(x, what) {
  mean(unlist(x$ctx$records$profile[[what]]))
}
