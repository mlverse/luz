#' @include callbacks.R
NULL

#' tfevents callback
#'
#' Logs metrics and other model information in the tfevents file format.
#' Assuming tensorboard is installed, result can be visualized with
#'
#' ```
#' tensorboard --logdir=logs
#' ```
#'
#' @param logdir A directory to where log will be written to.
#' @param histograms A boolean specifying if histograms of model weights should
#'   be logged. It can also be a character vector specifying the name of the parameters
#'   that should be logged (names are the same as `names(model$parameters)`).
#' @param ... Currently not used. For future expansion.
#'
#' @examples
#' if (torch::torch_is_installed()) {
#' library(torch)
#' x <- torch_randn(1000, 10)
#' y <- torch_randn(1000, 1)
#'
#' model <- nn_linear %>%
#'   setup(loss = nnf_mse_loss, optimizer = optim_adam) %>%
#'   set_hparams(in_features = 10, out_features = 1) %>%
#'   set_opt_hparams(lr = 1e-4)
#'
#' tmp <- tempfile()
#'
#' model %>% fit(list(x, y), valid_data = 0.2, callbacks = list(
#'   luz_callback_tfevents(tmp, histograms = TRUE)
#' ))
#' }
#' @export
luz_callback_tfevents <- luz_callback(
  name = "tfevents_callback",
  initialize = function(logdir = "logs", histograms = FALSE, ...) {
    rlang::check_installed(pkg = "tfevents")
    self$logdir <- logdir
    self$histograms <- histograms
  },
  log_histograms = function() {
    tfevents::local_logdir(self$logdir)
    parameters <- ctx$model$parameters

    if (is.character(self$histograms)) {
      parameters <- parameters[self$histograms]
    }

    histograms <- list()
    histograms[["weights"]] <- lapply(parameters, function(x) {
      tfevents::summary_histogram(as.array(x$cpu()))
    })

    tfevents::local_logdir(self$logdir)
    tfevents::log_event(!!!histograms, step = ctx$epoch)
  },
  log_events = function(set) {
    tfevents::local_logdir(self$logdir)
    metrics <- list()
    metrics[[set]] <- ctx$get_metrics(set = set, epoch = ctx$epoch)
    tfevents::log_event(!!!metrics, step = ctx$epoch)
  },
  on_train_end = function() {
    self$log_events("train")
    if (isTRUE(self$histograms) || is.character(self$histograms)) {
      self$log_histograms()
    }
  },
  on_valid_end = function() {
    self$log_events("valid")
  }
)
