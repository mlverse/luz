#' Saves luz objects to disk
#'
#' Allows saving luz fitted models to the disk. Objects can be loaded back with
#' [luz_load()].
#'
#' @note Objects are saved as plain `.rds` files but `obj$model` is serialized
#' with `torch_save` before saving it.
#'
#' @section Warning:
#' The [ctx] is naively serialized. Ie, we only use [saveRDS()] to serialize it.
#' Don't expect `luz_save` to work correctly if you have unserializable objects
#' in the [ctx] like `torch_tensor`s and external pointers in general.
#'
#' @param obj an object of class 'luz_module_fitted' as returned by
#' [fit.luz_module_generator()].
#' @param path path in file system to the object.
#' @param ... currently unused.
#'
#' @family luz_save
#' @export
luz_save <- function(obj, path, ...) {
  rlang::check_dots_empty()
  # dangling environments might be in the `obj` search path causing problems
  # during saving. `gc()` is a good practice to make sure they are cleaned up
  # before saving.
  gc()

  if (!inherits(obj, "luz_module_fitted"))
    rlang::abort("luz_save only works with 'luz_module_fitted_objects' and got {class(obj)[1]}")

  # avoid warning because luz will always be available when reloading
  # because we reload with `luz_load()`.
  suppressWarnings({
    serialized_model <- model_to_raw(obj$model)
    obj$ctx$.serialized_model <- serialized_model
    obj$ctx$.serialization_version <- 2L
    o <- saveRDS(obj, path)
  })

  invisible(o)
}



#' Load trained model
#'
#' Loads a fitted model. See documentation in [luz_save()].
#'
#' @inheritParams luz_save
#'
#' @family luz_save
#' @export
luz_load <- function(path) {
  obj <- readRDS(file = path)

  if (is.null(obj$ctx$.serialization_version))
    return(legacy_luz_load(path))

  model <- model_from_raw(obj$ctx$.serialized_model)

  obj$model <- model
  obj$ctx$.serialized_model <- NULL
  obj$ctx$.serialization_version <- NULL

  obj
}

legacy_luz_load <- function(path) {
  obj <- readRDS(file = path)
  model <- model_from_raw(obj$ctx$.serialized_model)
  bind_context(model, obj$ctx)
  obj$model <- model
  obj$ctx$model <- model
  rm(envir = obj$ctx, list = ".serialized_model")
  obj
}

#' Loads model weights into a fitted object.
#'
#' This can be useful when you have saved model checkpoints during training and
#' want to reload the best checkpoint in the end.
#'
#' @section Warning:
#' `luz_save_model_weights` operates inplace, ie modifies the model object to contain the
#' new weights.
#'
#' @returns
#' Returns `NULL` invisibly.
#'
#' @param obj luz object to which you want to copy the new weights.
#' @param path path to saved model in disk.
#' @param ... other arguments passed to [torch_load()].
#'
#' @export
luz_load_model_weights <- function(obj, path, ...) {
  saved_model <- torch::torch_load(path, ...)
  obj$model$load_state_dict(saved_model$state_dict())
  # we return NULL to make sure people don't expect it to return a copy of obj
  invisible(NULL)
}

#' @rdname luz_load_model_weights
#' @export
luz_save_model_weights <- function(obj, path) {
  # 'package:luz' may not be available when loading
  suppressWarnings({
    o <- torch::torch_save(obj$model, path)
  })
  invisible(o)
}

luz_checkpoint <- function(ctx, path) {
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

  torch_save(state, path)
}

#' Loads a checkpoint
#'
#' Works with checkpoints created typically with [luz_callback_model_checkpoint()].
#'
#' @param obj Object to which we want to load the checkpoint.
#' @param path Path of the checkpoint on disk.
#' @param ... unused. Is there to allow future extensions.
#'
#' @export
luz_load_checkpoint <- function(obj, path, ...) {
  UseMethod("luz_load_checkpoint")
}

#' @note This only modifies the model weights, not records, etc.
#' @export
luz_load_checkpoint.luz_module_fitted <- function(obj, path, ...) {
  state <- torch_load(path)
  # this modifies obj in place / records are not modified from the results
  obj$model$load_state_dict(state$model)
  invisible(NULL)
}

#' @inheritParams luz_callback_resume_from_checkpoint
#' @export
luz_load_checkpoint.luz_fit_context <- function(obj, path, ...,
                                                restore_records = TRUE,
                                                restore_optimizer_state = TRUE,
                                                restore_callbacks_state = TRUE,
                                                restore_model_state = TRUE
                                                ) {
  ctx <- obj # nicer name as we refer to fields in the ctx object.

  # load state dicts if they are available
  state <- to_device(torch_load(path), ctx$device)

  # load objects to their place.
  if (restore_model_state) {
    ctx$model$load_state_dict(state$model)
  }

  if (restore_optimizer_state) {
    map2(ctx$optimizers, state$optimizers, function(x, y) {
      x$load_state_dict(y)
    })
  }

  if (restore_records) {
    ctx$unsafe_set_records(state$records)
  }

  if (restore_callbacks_state) {
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
  }

  invisible(NULL)
}

model_to_raw <- function(model) {
  con <- rawConnection(raw(), open = "wr")
  torch::torch_save(model, con)
  on.exit({close(con)}, add = TRUE)
  r <- rawConnectionValue(con)
  r
}

model_from_raw <- function(object) {
  con <- rawConnection(object)
  on.exit({close(con)}, add = TRUE)
  module <- torch::torch_load(con)
  module
}
