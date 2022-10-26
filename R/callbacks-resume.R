#' Resume training callback
#'
#' This callback allows you to resume training a model.
#'
#' When using it, model weights, optimizer state are serialized at the end of
#' each epoch. If something fails during training simply re-running the same
#' script will restart the model training from the epoch right after the last
#' epoch that was serialized.
#'
#' @note In general you will want to add this callback as the last in the callbacks
#' list, this way, the serialized state is likely to contain all possible changes
#' that other callbacks could have made at `'on_epoch_end'`. The default `weight`
#' attribute of this callback is `Inf`.
#'
#' @section Customizing serialization:
#'
#' By default model, optimizer state and records are serialized. Callbacks can
#' be used to customize serialization by implementing the `state_dict()` and
#' `load_state_dict()` methods.
#' If those methods are implemented, then `state_dict()` is called at the end of
#' each epoch and `load_state_dict()` is called when the model is resumed.
#'
#' @note Read the checkpointing article in the pkgdown website for more
#'  information.
#'
#' @param path Path to save state files for the model.
#'
#' @family luz_callbacks
#'
#' @examples
#' if (torch::torch_is_installed()) {
#' library(torch)
#' library(luz)
#'
#' x <- torch_randn(1000, 10)
#' y <- torch_randn(1000, 1)
#'
#' model <- nn_linear %>%
#'   setup(optimizer = optim_sgd, loss = nnf_mse_loss) %>%
#'   set_hparams(in_features = 10, out_features = 1) %>%
#'   set_opt_hparams(lr = 0.01)
#'
#'
#' # simulate a failure in the middle of epoch 5 happening only once.
#' callback_stop <- luz_callback(
#'   "interrupt",
#'   failed = FALSE,
#'   on_epoch_end = function() {
#'     if (ctx$epoch == 5 && !self$failed) {
#'       self$failed <- TRUE
#'       stop("Error on epoch 5")
#'     }
#'   }
#' )
#'
#' path <- tempfile()
#' autoresume <- luz_callback_auto_resume(path = path)
#' interrupt <- callback_stop()
#'
#' # try once and the model fails
#' try({
#'   results <- model %>% fit(
#'     list(x, y),
#'     callbacks = list(autoresume, interrupt),
#'     verbose = FALSE
#'   )
#' })
#'
#' # model resumes and completes
#' results <- model %>% fit(
#'   list(x, y),
#'   callbacks = list(autoresume, interrupt),
#'   verbose = FALSE
#' )
#'
#' get_metrics(results)
#'
#' }
#' @export
luz_callback_auto_resume <- luz_callback(
  "auto_resume_callback",
  weight = Inf,
  initialize = function(path = "./state.pt") {
    self$path <- file.path(path)
    fs::dir_create(fs::path_dir(self$path), recurse = TRUE)
  },
  on_fit_begin = function() {
    ctx$epoch_handlers <- append(
      ctx$epoch_handlers,
      # this condition will only break the epoch execution and go to the next epoch
      list(break_epoch = function(e) {})
    )

    # we make no assertion that the model is the same when reloading stuff.
    # it's user's responsability to make sure that files contained in `self$path`
    # are compatible with the model.
    if (!file.exists(self$path)) {
      return(invisible(NULL))
    }

    if (ctx$verbose) {
      cli::cli_inform(c(
        "!" = "Resuming model training with of {.fun luz_callback_auto_resume}",
        "i" = "State will be reloaded from: {.file {self$path}}"
      ))
    }

    luz_load_checkpoint(ctx, self$path)
  },
  state_dict = function() {
    list(current_epoch = ctx$epoch)
  },
  load_state_dict = function(dict) {
    self$current_epoch <- dict$current_epoch
  },
  on_epoch_begin = function() {
    if (is.null(self$current_epoch)) return(invisible(NULL))
    if (self$current_epoch >= ctx$epoch) {
      rlang::signal("Break epoch", class = "break_epoch")
    }
  },
  on_epoch_end = function() {
    luz_checkpoint(ctx, self$path)
  },
  on_fit_end = function() {
    fs::file_delete(self$path)
    self$current_epoch <- NULL
  }
)

#' Allow resume model training from a specific checkpoint
#'
#' @param path Path to the checkpoint that you want to resume.
#' @param restore_model_state Wether to restore the model state from the callback.
#' @param restore_records Wether to restore records from the checkpoint.
#' @param restore_optimizer_state Wether to restore the optimizer state from the
#'   checkpoint.
#' @param restore_callback_state Wether to restore the callbacks state from the
#'   checkpoint.
#'
#' @note Read the checkpointing article in the pkgdown website for more
#'  information.
#'
#' @seealso [luz_callback_model_checkpoint()]
#' @family luz_callbacks
#'
#' @export
luz_callback_resume_from_checkpoint <- luz_callback(
  "resume_from_checkpoint_callback",
  initialize = function(path, ...,
                        restore_model_state = TRUE,
                        restore_records = FALSE,
                        restore_optimizer_state = FALSE,
                        restore_callbacks_state = FALSE) {

    ellipsis::check_dots_empty()

    # if path is a directory, grab the last file returned by dir_ls
    if (fs::is_dir(path)) {
      path <- tail(fs::dir_ls(path), 1)
    }

    self$path <- path

    if ((length(self$path) == 0) || (!fs::file_exists(self$path))) {
      cli::cli_warn(c(
        "The checkpoint path {.file {self$path}} does not exist.",
        i = "Will start with state initialized with default methods."
      ))
      self$path_exists <- FALSE
    } else {
      self$path_exists <- TRUE
    }

    self$params <- list(
      restore_records = restore_records,
      restore_optimizer_state = restore_optimizer_state,
      restore_callbacks_state = restore_callbacks_state,
      restore_model_state = restore_model_state
    )
  },
  on_fit_begin = function() {
    if (self$path_exists) {
      rlang::exec(luz_load_checkpoint, obj = ctx, path = self$path, !!!self$params)
    }
  }
)
