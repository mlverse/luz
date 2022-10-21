#' Resume training callback
#'
#' This callback allows you to resume training a model from a specified
#' directory.
#'
#' When using it, model weights, optimizer state and learning rate schedulers
#' state is serialized each epoch. If something fails during training simply
#' re-running the same script will restart the model training from where it
#' stopped.
#'
#' @param path Path to save state files for the model.
#'
#' @export
luz_callback_auto_resume <- luz_callback(
  "auto_resume_callback",
  initialize = function(path = "./state") {
    self$path <- file.path(path, "state.pt")
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
    state <- torch_load(self$path)

    # load objects to their place.
    ctx$model$load_state_dict(state$model)
    map2(ctx$optimizers, state$optimizers, function(x, y) {
      x$load_state_dict(y)
    })
    ctx$unsafe_set_records(state$records)

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

    # others?
    # TODO think of a way to allowusers to save other stuff here? maybe traverse
    # the callbacks looking for a state_dict method?

    torch_save(state, self$path)
  },
  on_fit_end = function() {
    fs::file_delete(self$path)
    self$current_epoch <- NULL
  }
)
