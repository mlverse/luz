
#' Set's up a `nn_module` to use with luz
#'
#' The setup function is used to set important attributes and method for `nn_modules`
#' to be used with Luz.
#'
#' It makes sure the module have all the necessary ingredients in order to be fitted.
#'
#' @param module (`nn_module`) The `nn_module` that you want set up.
#' @param loss (`function`, optional) An optional function with the signature
#' `function(input, target)`. It's only requires if your `nn_module` doesn't
#' implement a method called `loss`.
#' @param optimizer (`torch_optimizer`, optional) A function with the signature
#' `function(parameters, ...)` that is used to initialize an optimizer given
#' the model parameters.
#' @param metrics (`list`, optional) A list of metrics to be tracked during
#' the training procedure.
#'
#' @family training
#'
#' @export
setup <- function(module, loss = NULL, optimizer = NULL, metrics = NULL) {

  methods <- list()

  if (!is.null(loss))
    methods$loss <- function(input, target) {
      loss(input, target)
    }
  else if (!has_method(module, "loss") && !has_method(module, "step"))
    rlang::abort(c("No loss function has been provided.",
                   "Use the `loss` argument or,",
                   "Implement the `loss` method in the `nn_module` or,",
                   "Implement a custom `step` method that manually optimized the parameters."))

  if (!is.null(optimizer))
    methods$optimizer <- function(...) {
      optimizer(self$parameters, ...)
    }
  else if (!has_method(module, "optimizer"))
    rlang::abort(c("No optimizer definition has been provided.",
                   "Use the optimizer argument or,",
                   "Implement the `optimizer` method in the `nn_module`."))

  metrics <- c(luz_metric_loss_average, metrics)
  methods$metrics <- metrics

  if (!has_forward_method(module))
    methods$forward <- identity

  mod <- do.call(
    torch::nn_module,
    append(methods, list(name = "luz_module", inherit = module))
  )
  class(mod) <- c("luz_module_generator")
  mod
}

set_hparams <- function(module, ...) {
  hparams <- rlang::list2(...)
  attr(module, "hparams") <- hparams
  module
}

get_hparams <- function(module) {
  attr(module, "hparams")
}

#' @importFrom generics fit
#' @export
fit.luz_module_generator <- function(module, data, epochs = 10, callbacks = NULL,
                                     valid_data = NULL, accelerator = NULL,
                                     verbose = NULL) {

  # Initialize context:
  ctx <- rlang::new_environment()

  if (is.null(verbose)) {
    ctx$verbose <- interactive()
  } else {
    ctx$verbose <- verbose
  }

  if (is.null(accelerator))
    accelerator <- accelerator()

  ctx$accelerator <- accelerator

  model <- do.call(module, get_hparams(module) %||% list())
  bind_context(model, ctx)

  optimizers <- do.call(model$optimizer, get_hparams(module)$opt_hparams %||% list())

  if (!is.list(optimizers)) {
    optimizers <- list(opt = optimizers)
  }

  if (!rlang::is_named(optimizers)) {
    rlang::abort(c("List of optimizers is not named.",
                   "When returning a list of optimizers, the list must be named."))
  }

  c(model, optimizers, data, valid_data) %<-%
    ctx$accelerator$prepare(model, optimizers, data, valid_data)

  ctx$model <- model
  ctx$model$ctx <- ctx

  ctx$optimizers <- optimizers
  ctx$data <- data
  ctx$valid_data <- valid_data

  ctx$epochs <- epochs
  callbacks <- append(default_callbacks(), callbacks)
  ctx$callbacks <- initialize_callbacks(callbacks, ctx)

  if (is.null(ctx$model$step))
    step <- function() default_step(ctx)
  else
    step <- ctx$model$step

  ctx$call_callbacks <- function(name) {
    call_all_callbacks(ctx$callbacks, name)
  }

  ctx$call_callbacks("on_fit_begin")

  for (epoch in seq_len(ctx$epochs)) {
    ctx$epoch <- epoch
    ctx$iter <- 0L
    ctx$call_callbacks("on_epoch_begin")

    ctx$call_callbacks("on_train_begin")

    coro::loop(for (batch in ctx$data) {
      bind_batch_to_ctx(ctx, batch)
      ctx$iter <- ctx$iter + 1L

      ctx$call_callbacks("on_train_batch_begin")
      step()
      ctx$call_callbacks("on_train_batch_end")
    })

    ctx$call_callbacks("on_train_end")

    if (!is.null(ctx$valid_data)) {

      ctx$call_callbacks("on_valid_begin")

      ctx$iter <- 0L
      torch::with_no_grad({
        coro::loop(for (batch in ctx$valid_data) {
          bind_batch_to_ctx(ctx, batch)
          ctx$iter <- ctx$iter + 1L

          ctx$call_callbacks("on_valid_batch_begin")
          step()
          ctx$call_callbacks("on_valid_batch_end")
        })
      })

      ctx$call_callbacks("on_valid_end")

    }

    ctx$call_callbacks("on_epoch_end")
  }

  ctx$call_callbacks("on_fit_end")
  structure(
    list(
      model  = ctx$model,
      losses = ctx$losses,
      record = ctx$records,
      ctx = ctx
    ),
    class = "luz_module_fitted"
  )
}

#' @importFrom stats predict
#' @export
predict.luz_module_fitted <- function(object, newdata, ..., callbacks = list(),
                                       accelerator = NULL) {

  ctx <- object$ctx

  if (is.null(accelerator))
    accelerator <- accelerator()

  ctx$accelerator <- accelerator
  model <- NULL; data <- NULL
  c(model, data) %<-% ctx$accelerator$prepare(ctx$model, ctx$data)

  ctx$model <- model
  ctx$data <- newdata

  ctx$model$eval()
  ctx$training <- FALSE

  pars <- rlang::list2(...)
  if (is.null(pars$stack))
    stack <- TRUE
  else
    stack <- pars$stack

  ctx$output <- list()

  ctx$callbacks <- initialize_callbacks(callbacks, ctx)

  torch::with_no_grad({
    ctx$call_callbacks("on_predict_begin")
    coro::loop(for(batch in data) {
      ctx$batch <- batch
      ctx$input <- batch[[1]]
      ctx$call_callbacks("on_predict_batch_begin")
      ctx$output[[length(ctx$output) + 1]] <- do.call(ctx$model, list(ctx$input))
      ctx$call_callbacks("on_predict_batch_end")
    })
    ctx$call_callbacks("on_predict_end")
  })

  if (stack) {
    ctx$output <- torch::torch_stack(ctx$output)
  }

  ctx$output
}

bind_batch_to_ctx <- function(ctx, batch) {
  ctx$batch <- batch
  ctx$input <- ctx$batch[[1]]
  ctx$target <- ctx$batch[[2]]
}

default_step <- function(ctx) {
  if (ctx$training)
    fit_one_batch(ctx)
  else
    valid_one_batch(ctx)
}

fit_one_batch <-function(ctx) {
  for (nm in names(ctx$optimizers)) {
    ctx$pred <- do.call(ctx$model, list(ctx$input))
    ctx$call_callbacks("on_train_batch_after_pred")

    ctx$opt <- ctx$optimizers[[nm]]
    ctx$opt_name <- nm

    ctx$loss_grad <- ctx$model$loss(ctx$pred, ctx$target)
    ctx$loss[[ctx$opt_name]] <- ctx$loss_grad$detach()

    ctx$call_callbacks("on_train_batch_after_loss")

    ctx$call_callbacks("on_train_batch_before_backward")
    ctx$loss_grad$backward()

    ctx$call_callbacks("on_train_batch_before_step")
    ctx$opt$step()
    ctx$opt$zero_grad()
    ctx$call_callbacks("on_train_batch_after_step")
  }
}

valid_one_batch <- function(ctx) {
  for (nm in names(ctx$optimizers)) {
    ctx$opt_name <- nm

    ctx$pred <- do.call(ctx$model, list(ctx$input))
    ctx$call_callbacks("on_valid_batch_after_pred")

    ctx$loss[[ctx$opt_name]] <- ctx$model$loss(ctx$pred, ctx$target)
    ctx$call_callbacks("on_valid_batch_after_loss")
  }
}

initialize_callbacks <- function(callbacks, ctx) {
  lapply(callbacks, function(cb) {
    cb$set_ctx(ctx)
    bind_context(cb, ctx)
    cb
  })
}
