
light_module <- function(module, loss = NULL, optimizer = NULL, metrics = NULL) {

  if (!is.null(loss))
    loss_fn <- function(input, target) {
      loss(input, target)
    }

  if (!is.null(optimizer))
    opt_fn <- function(...) {
      optimizer(self$parameters, ...)
    }

  torch::nn_module(
    "light_module",
    inherit = module,
    loss = loss_fn,
    optimizer = opt_fn,
    metrics = metrics
  )
}

set_hparams <- function(module, ...) {
  hparams <- rlang::list2(...)
  attr(module, "hparams") <- hparams
  module
}

get_hparams <- function(module) {
  attr(module, "hparams")
}


fit <- function(module, data, epochs = 10, callbacks = NULL, valid_data = NULL,
                accelerator = NULL) {

  # Initialize context:
  ctx <- rlang::new_environment()

  if (is.null(accelerator))
    accelerator <- accelerator()

  ctx$accelerator <- accelerator

  model <- do.call(module, get_hparams(module) %||% list())
  opt <- do.call(model$optimizer, get_hparams(module)$opt_hparams %||% list())

  c(model, opt, data, valid_data) %<-%
    ctx$accelerator$prepare(model, opt, data, valid_data)

  ctx$model <- model
  ctx$opt <- opt
  ctx$data <- data
  ctx$valid_data <- valid_data

  ctx$epochs <- epochs
  callbacks <- append(default_callbacks(), callbacks)
  ctx$callbacks <- lapply(callbacks, function(cb) {
    cb$new(ctx = ctx)
  })

  ctx$call_callbacks <- function(name) {
    call_all_callbacks(ctx$callbacks, name)
  }

  ctx$call_callbacks("on_fit_begin")

  for (epoch in seq_len(ctx$epochs)) {
    ctx$epoch <- epoch
    ctx$call_callbacks("on_epoch_begin")

    ctx$call_callbacks("on_train_begin")

    ctx$model$train()

    coro::loop(for (batch in ctx$data) {

      ctx$batch <- batch
      ctx$input <- ctx$batch[[1]]
      ctx$target <- ctx$batch[[2]]

      ctx$call_callbacks("on_train_batch_begin")

      fit_one_batch(ctx)

      ctx$call_callbacks("on_train_batch_end")
    })

    ctx$call_callbacks("on_train_end")
    ctx$call_callbacks("on_valid_begin")

    ctx$model$eval()
    with_no_grad({
      coro::loop(for (batch in ctx$valid_data) {
        ctx$batch <- batch
        ctx$input <- ctx$batch[[1]]
        ctx$target <- ctx$batch[[2]]
        ctx$call_callbacks("on_valid_batch_begin")
        valid_one_batch(ctx)
        ctx$call_callbacks("on_valid_batch_end")
      })
    })

    ctx$call_callbacks("on_valid_end")
    ctx$call_callbacks("on_epoch_end")
  }

  ctx$call_callbacks("on_fit_end")
  ctx$model
}

fit_one_batch <-function(ctx) {

  ctx$pred <- do.call(ctx$model, list(ctx$input))

  ctx$call_callbacks("on_train_batch_after_pred")

  ctx$loss_grad <- ctx$model$loss(ctx$pred, ctx$target)
  ctx$loss <- ctx$loss_grad$detach()

  ctx$call_callbacks("on_train_batch_after_loss")

  ctx$call_callbacks("on_train_batch_before_backward")

  ctx$loss_grad$backward()

  ctx$call_callbacks("on_train_batch_before_step")
  ctx$opt$step()
  ctx$opt$zero_grad()

  ctx$call_callbacks("on_train_batch_after_step")
}

valid_one_batch <- function(ctx) {
  ctx$pred <- do.call(ctx$model, list(ctx$input))
  ctx$call_callbacks("on_valid_batch_after_pred")

  ctx$loss <- ctx$model$loss(ctx$pred, ctx$target)
  ctx$call_callbacks("on_valid_batch_after_loss")
}
