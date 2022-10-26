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
#' @param path Path to save state files for the model.
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

    # load state dicts if they are available
    state <- to_device(torch_load(self$path), ctx$device)

    # load objects to their place.
    ctx$model$load_state_dict(state$model)
    map2(ctx$optimizers, state$optimizers, function(x, y) {
      x$load_state_dict(y)
    })


    ctx$unsafe_set_records(state$records)

    # reload callbacks state_dicts
    map2(state$callbacks, ctx$callbacks, function(st, cb) {
      if (is.null(st)) return()

      if (is.null(cb$load_state_dict) && !is.null(st)) {
        cli::cli_abort(c(
          x = "Failed resuming the model.",
          i = paste0(
            "A callback with class {.cls {class(cb)} has state attached ",
            "to it, but doesn't implement the {.fn load_state_dict} method."
          )
        ))
      }

      cb$load_state_dict(st)
    })

    self$current_epoch <- state$epoch

  },
  on_epoch_begin = function() {
    if (is.null(self$current_epoch)) return(invisible(NULL))
    if (self$current_epoch >= ctx$epoch) {
      rlang::signal("Break epoch", class = "break_epoch")
    }
  },
  on_epoch_end = function() {
    state <- list()

    #grab epoch
    state[["epoch"]] <- ctx$epoch
    state[["records"]] <- ctx$records

    # grab model state
    state[["model"]] <- ctx$model$state_dict()

    # grab optimizer state
    state[["optimizers"]] <- lapply(ctx$optimizers, function(x) x$state_dict())

    # traverse callbacks looking for the `state_dict()` method.
    state[["callbacks"]] <- lapply(ctx$callbacks, function(x) {
      if (is.null(x$state_dict))
        NULL
      else
        x$state_dict()
    })

    torch_save(state, self$path)
  },
  on_fit_end = function() {
    fs::file_delete(self$path)
    self$current_epoch <- NULL
  }
)
