
#' Traces the model for training and validation to get a few speedups.
#'
#'
#' @export
luz_callback_trace <- luz_callback(
  "trace_callback",
  initialize = function(check_train = TRUE, check_valid = TRUE) {
    self$traced_models <- list(train = NULL, valid = NULL)
    self$check_train <- check_train
    self$check_valid <- check_valid
  },
  on_train_batch_after_pred = function() {

    if (ctx$epoch > 1 || ctx$iter > 1) return()

    # Traces the model for training
    if (is.null(self$traced_models$train)) {
      traced <- torch::jit_trace_module(
        ctx$model,
        forward = ctx$input,
        loss = list(ctx$pred, ctx$target)
      )
    } else {
      traced <- self$traced_models$train
      rlang::warn("The same traced model is going to be used both optimizers and this might lead to unexpected results.")
    }

    if (self$check_train) {

      equal <- all.equal(do.call(traced, list(ctx$input)), ctx$pred)
      if (!equal) {
        rlang::abort(
          "Traced model didn't produce identical results when compared to the original model during training.",
          "Are you sure that tracing is producing the correct output? If yes, disable this error with `check_train=FALSE`."
        )
      }
    }

    self$traced_models$train <- traced
    ctx$model <- self$traced_models$train
  },
  on_train_begin = function() {
    self$model <- ctx$model
    if (ctx$epoch <= 1) return()
    ctx$model <- self$traced_models$train
  },
  on_train_end = function() {
    ctx$model <- self$model
    ctx$model$load_state_dict(self$traced_models$train$state_dict())
  },
  on_valid_batch_after_pred = function() {
    if (ctx$epoch > 1 || ctx$iter > 1) return()

    # Traces the model for validation
    if (is.null(self$traced_models$valid)) {
      traced <- torch::jit_trace_module(
        ctx$model,
        forward = ctx$input,
        loss = list(ctx$pred, ctx$target)
      )
    } else {
      traced <- self$traced_models$valid
      rlang::warn("The same traced model is going to be used both optimizers and this might lead to unexpected results.")
    }

    if (self$check_valid) {
      equal <- all.equal(do.call(traced, list(ctx$input)), ctx$pred)
      if (!equal) {
        rlang::abort(
          "Traced model didn't produce identical results when compared to the original model during validation",
          "Are you sure that tracing is producing the correct output? If yes, disable this error with `check_valid=FALSE`."
        )
      }
    }

    self$traced_models$valid <- traced
    ctx$model <- self$traced_models$valid
  },
  on_valid_begin = function() {
    self$model <- ctx$model
    if (ctx$epoch <= 1) return()
    ctx$model <- self$traced_models$valid
  },
  on_valid_end = function() {
    ctx$model <- self$model
  }
)

all.equal.torch_tensor <- function(target, current, ...) {
  torch::torch_allclose(target, current, ...)
}
