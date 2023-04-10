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
#'
#' @note In general you don't need to use these callback by yourself because it's always
#'   included by default in [fit.luz_module_generator()].
#'
#' @examples
#' profile_callback <- luz_callback_profile()
#'
#' @returns
#' A `luz_callback`
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

  on_valid_begin = function() {
    self$tic("valid")
  },
  on_valid_end = function() {
    ctx$log("profile", "valid", self$toc("valid"))
  },

  tic = function(name) {
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
  unlist(x$records$profile$fit)
}

get_average_time <- function(x, what) {
  mean(unlist(x$records$profile[[what]]))
}
