
light_module <- function(module, loss = NULL, optimizer = NULL, metrics = NULL) {


  methods <- list()
  methods$metrics <- metrics

  if (!is.null(loss))
    methods$loss <- function(input, target) {
      loss(input, target)
    }
  else if (!has_method(module, "loss"))
    rlang::abort(c("No loss function has been provided.",
                   "Use the `loss` argument or,",
                   "Implement the `loss` method in the `nn_module`."))

  if (!is.null(optimizer))
    methods$optimizer <- function(...) {
      optimizer(self$parameters, ...)
    }
  else if (!has_method(module, "optimizer"))
    rlang::abort(c("No optimizer definition has been provided.",
                   "Use the optimizer argument or,",
                   "Implement the `optimizer` method in the `nn_module`."))

  if (!has_forward_method(module))
    methods$forward <- identity

  do.call(torch::nn_module,
          append(methods, list(name = "light_module", inherit = module)))
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

    coro::loop(for (batch in ctx$data) {

      bind_batch_to_ctx(ctx, batch)

      ctx$call_callbacks("on_train_batch_begin")
      fit_one_batch(ctx)
      ctx$call_callbacks("on_train_batch_end")

    })

    ctx$call_callbacks("on_train_end")
    ctx$call_callbacks("on_valid_begin")

    with_no_grad({
      coro::loop(for (batch in ctx$valid_data) {

        bind_batch_to_ctx(ctx, batch)

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

bind_batch_to_ctx <- function(ctx, batch) {
  ctx$batch <- batch
  ctx$input <- ctx$batch[[1]]
  ctx$target <- ctx$batch[[2]]
}

fit_one_batch <-function(ctx) {
  ctx$pred <- do.call(ctx$model, list(ctx$input))
  ctx$call_callbacks("on_train_batch_after_pred")

  for (nm in names(ctx$optimizers)) {
    ctx$opt <- ctx$optimizers[[nm]]
    ctx$opt_name <- nm

    ctx$loss_grad <- ctx$model$loss(ctx$pred, ctx$target)
    ctx$loss <- ctx$loss_grad$detach()
    ctx$call_callbacks("on_train_batch_after_loss")

    ctx$call_callbacks("on_train_batch_before_backward")
    ctx$loss_grad$backward()

    ctx$call_callbacks("on_train_batch_before_step")
    ctx$opt$step()
    ctx$opt$zero_grad()
  }

  ctx$call_callbacks("on_train_batch_after_step")

}

valid_one_batch <- function(ctx) {
  for (nm in names(ctx$optimizers)) {
    ctx$opt_name <- nm

    ctx$pred <- do.call(ctx$model, list(ctx$input))
    ctx$call_callbacks("on_valid_batch_after_pred")

    ctx$loss <- ctx$model$loss(ctx$pred, ctx$target)
    ctx$call_callbacks("on_valid_batch_after_loss")
  }
}
