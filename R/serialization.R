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
#' @param path path in file system so save the object.
#' @param ... currently unused.
#'
#' @family luz_save
#' @export
luz_save <- function(obj, path, ...) {
  ellipsis::check_dots_empty()

  if (!inherits(obj, "luz_module_fitted"))
    rlang::abort("luz_save only works with 'luz_module_fitted_objects' and got {class(obj)[1]}")

  # avoid warning because luz will always be available when reloading
  # because we reload with `luz_load()`.
  suppressWarnings({
    serialized_model <- model_to_raw(obj$ctx$model)
    obj$ctx$.serialized_model <- serialized_model
    o <- saveRDS(obj, path)
  })

  o
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
#'
#' @export
luz_load_model_weights <- function(obj, path) {
  saved_model <- torch::torch_load(path)
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
