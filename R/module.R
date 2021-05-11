
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
    methods$set_optimizers <- function(...) {
      optimizer(self$parameters, ...)
    }
  else if (!has_method(module, "set_optimizers"))
    rlang::abort(c("No optimizer definition has been provided.",
                   "Use the optimizer argument or,",
                   "Implement the `set_optimizers` method in the `nn_module`."))

  metrics <- c(luz_metric_loss_average(), metrics)
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

#' Set hyper-parameter of a module
#'
#' @description
#' This function is used to define hyper-parameters before calling `fit` for
#' `luz_modules`.
#'
#' @param module An `nn_module` that has been [setup()].
#' @param ... The parameters set here will be used to initialize the `nn_module`, ie they
#' are passed unchanged to the `initialize` method of the base `nn_module`.
#'
#' @family set_hparam
#'
#' @export
set_hparams <- function(module, ...) {
  hparams <- rlang::list2(...)
  attr(module, "hparams") <- hparams
  module
}

#' Set optimizer hyper-parameters
#'
#' @description
#' This function is used to define hyper-parameters for the optimizer initialization
#' method.
#'
#' @inheritParams set_hparams
#' @param ... The parameters passed here will be used to initialize the optimizers.
#' For example, if your optimizer is `optim_adam` and you pass `lr=0.1`, then the
#' `optim_adam` function is called with `optim_adam(parameters, lr=0.1)` when fitting
#' the model.
#'
#' @family set_hparam
#' @export
set_opt_hparams <- function(module, ...) {
  hparams <- rlang::list2(...)
  attr(module, "opt_hparams") <- hparams
  module
}

get_hparams <- function(module) {
  attr(module, "hparams")
}

get_opt_hparams <- function(module) {
  attr(module, "opt_hparams")
}

#' Fit a `nn_module`
#'
#' @param object An `nn_module` that has been [setup()].
#'
#' @param data (dataloader) A dataloader created with [torch::dataloader()] used
#' for training the model. The dataloader must return a list with at most 2 items.
#' The first item will be used as input for the module and the second will be used
#' as target for the loss function.
#'
#' @param epochs (int) The number of epochs for training the model.
#'
#' @param callbacks (list, optional) A list of callbacks defined with [luz_callback()] that
#' will be called during the training procedure. The callbacks [luz_callback_metrics()],
#' [luz_callback_progress()] and [luz_callback_train_valid()] are always added by default.
#'
#' @param valid_data (dataloader, optional) A dataloader created with [torch::dataloader()]
#' that will be used during the validation procedure.
#'
#' @param accelerator (accelerator, optional) An optional [accelerator()] object used
#' to configure device placement of the components like [nn_module]s, optimizers
#' and batches of data.
#'
#' @param verbose (logical, optional) An optional boolean value indicating if the
#' fitting procedure should emmit output to the console during training. By default,
#' it will produce output if [interactive()] is `TRUE`, otherwise it won't print
#' to the console.
#'
#' @param ... Currently unused,
#'
#' @importFrom generics fit
#' @export
#'
fit.luz_module_generator <- function(object, data, epochs = 10, callbacks = NULL,
                                     valid_data = NULL, accelerator = NULL,
                                     verbose = NULL, ...) {

  module <- object
  ellipsis::check_dots_empty()

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

  optimizers <- do.call(model$set_optimizers, get_opt_hparams(module) %||% list())

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

  ctx$handlers <- list()

  ctx$call_callbacks("on_fit_begin")
  rlang::with_handlers(
    !!! ctx$handlers,
    .expr = {
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
    })

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

  ctx$handlers <- list()
  ctx$output <- list()
  ctx$callbacks <- initialize_callbacks(callbacks, ctx)

  predict_fn <- if (is.null(ctx$model$predict)) ctx$model else ctx$model$predict

  torch::with_no_grad({
    ctx$call_callbacks("on_predict_begin")
    rlang::with_handlers(
      !!! ctx$handlers,
      .expr = {
        coro::loop(for(batch in data) {
          ctx$batch <- batch
          ctx$input <- batch[[1]]
          ctx$call_callbacks("on_predict_batch_begin")
          ctx$output[[length(ctx$output) + 1]] <- do.call(predict_fn, list(ctx$input))
          ctx$call_callbacks("on_predict_batch_end")
        })
      }
    )
    ctx$call_callbacks("on_predict_end")
  })

  if (stack) {
    ctx$output <- torch::torch_vstack(ctx$output)
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

#' Context object
#'
#' Context objects used in luz to share information between model methods,
#' metrics and callbacks.
#'
#' @name ctx
#'
#' @includeRmd man/rmd/ctx.Rmd details
#' @rdname ctx
NULL
