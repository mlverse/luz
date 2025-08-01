
#' Set's up a `nn_module` to use with luz
#'
#' The setup function is used to set important attributes and method for `nn_modules`
#' to be used with luz.
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
#' the training procedure. Sometimes, you want some metrics to be evaluated
#' only during training or validation, in this case you can pass a [luz_metric_set()]
#' object to specify metrics used in each stage.
#' @param backward (`function`) A functions that takes the loss scalar values as
#' it's parameter. It must call `$backward()` or [torch::autograd_backward()].
#' In general you don't need to set this parameter unless you need to customize
#' how luz calls the `backward()`, for example, if you need to add additional
#' arguments to the backward call. Note that this becomes a method of the `nn_module`
#' thus can be used by your custom `step()` if you override it.
#'
#' @note
#' It also adds a `device` active field that can be used to query the current
#' module `device` within methods, with eg `self$device`. This is useful when
#' [ctx()] is not available, eg, when calling methods from outside the `luz`
#' wrappers. Users can override the default by implementing a `device` active
#' method in the input `module`.
#'
#' @returns
#' A luz module that can be trained with [fit()].
#'
#' @family training
#'
#' @import torch
#' @export
setup <- function(module, loss = NULL, optimizer = NULL, metrics = NULL,
                  backward = NULL) {

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


  if (!is.null(backward)) {
    if(!rlang::is_function(backward)) {
      rlang::abort(c("backward should be a function with a single argument"))
    }
    methods$backward <- backward
  } else {
    methods$backward <- function(x) {
      x$backward()
    }
  }

  methods$metrics <- if (is_luz_metric_set(metrics)) {
    metrics
  } else {
    luz_metric_set(metrics)
  }

  # adds a device method, allowing users to quickly query the current
  # model device. this returns the device of the first parameter. should
  # be OK to do it, as users there's current no support for multi-gpu
  # training. users can override by implementing their own device method.
  if (is.null(get_method(module, "device"))) {
    methods$active <- list(
      device = function() {
        self$parameters[[1]]$device
      }
    )
  }

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
#' @returns
#' The same luz module
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
#' @returns
#' The same luz module
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
#' @param data (dataloader, dataset or list) A dataloader created with
#'   [torch::dataloader()] used for training the model, or a dataset created
#'   with [torch::dataset()] or a list. Dataloaders and datasets must return a
#'   list with at most 2 items. The first item will be used as input for the
#'   module and the second will be used as a target for the loss function.
#'
#' @param epochs (int) The maximum number of epochs for training the model. If a
#'   single value is provided, this is taken to be the `max_epochs` and
#'   `min_epochs` is set to 0. If a vector of two numbers is provided, the first
#'   value is `min_epochs` and the second value is `max_epochs`. The minimum and
#'   maximum number of epochs are included in the context object as
#'   `ctx$min_epochs` and `ctx$max_epochs`, respectively.
#'
#' @param callbacks (list, optional) A list of callbacks defined with
#'   [luz_callback()] that will be called during the training procedure. The
#'   callbacks [luz_callback_metrics()], [luz_callback_progress()] and
#'   [luz_callback_train_valid()] are always added by default.
#'
#' @param valid_data (dataloader, dataset, list or scalar value; optional) A
#'   dataloader created with [torch::dataloader()] or a dataset created with
#'   [torch::dataset()] that will be used during the validation procedure. They
#'   must return a list with (input, target). If `data` is a torch dataset or a
#'   list, then you can also supply a numeric value between 0 and 1 - and in
#'   this case a random sample with size corresponding to that proportion from
#'   `data` will be used for validation.
#'
#' @param accelerator (accelerator, optional) An optional [accelerator()] object
#'   used to configure device placement of the components like [torch::nn_module]s,
#'   optimizers and batches of data.
#'
#' @param verbose (logical, optional) An optional boolean value indicating if
#'   the fitting procedure should emit output to the console during training.
#'   By default, it will produce output if [interactive()] is `TRUE`, otherwise
#'   it won't print to the console.
#'
#' @param ... Currently unused.
#'
#' @param dataloader_options Options used when creating a dataloader. See
#'   [torch::dataloader()]. `shuffle=TRUE` by default for the training data and
#'   `batch_size=32` by default. It will error if not `NULL` and `data` is
#'   already a dataloader.
#'
#' @returns A fitted object that can be saved with [luz_save()] and can be
#' printed with [print()] and plotted with [plot()].
#'
#' @seealso [predict.luz_module_fitted()] for how to create predictions.
#'   [setup()] to find out how to create modules that can be trained with `fit`.
#'
#' @family training
#'
#' @importFrom generics fit
#'
#' @importFrom coro is_exhausted
#'
#' @export
fit.luz_module_generator <- function(
    object,
    data,
    epochs = 10,
    callbacks = NULL,
    valid_data = NULL,
    accelerator = NULL,
    verbose = NULL,
    ...,
    dataloader_options = NULL
) {

  enable_mps_fallback()
  module <- object
  rlang::check_dots_empty()

  # Initialize context:
  ctx <- fit_context$new(
    verbose = verbose,
    accelerator = accelerator,
    module = module,
    data = data,
    valid_data = valid_data,
    epochs = epochs,
    callbacks = callbacks,
    dataloader_options = dataloader_options
  )

  step <- get_step(ctx)

  # The environment of this function is leaking due to a bug
  # see https://github.com/mlverse/luz/issues/74
  # Until its fixed we clean up its environment so we don't keep
  # large objects here more than necessary.
  on.exit({
    e <- rlang::current_env()
    rm(list = rlang::env_names(e), envir = e)
  }, add = TRUE)

  ctx$call_callbacks("on_fit_begin")
  with_handlers(
    !!! ctx$handlers,
    .expr = {
      for (epoch in seq_len(ctx$max_epochs)) {
        with_handlers(
          !!! ctx$epoch_handlers,
          .expr = {

            ctx$epoch <- epoch
            ctx$iter <- 0L

            ctx$data <- ctx$train_data

            ctx$call_callbacks("on_epoch_begin")
            ctx$call_callbacks("on_train_begin")

            # this helps making sure the dataloader workers can be cleaned up
            # before the validation loop even starts.
            local({
              next_batch <- as_iterator(ctx$data)
              while(!is_exhausted(batch <- next_batch())) {
                ctx$batch<- batch
                ctx$iter <- ctx$iter + 1L

                ctx$call_callbacks("on_train_batch_begin")
                step()
                ctx$call_callbacks("on_train_batch_end")
              }
            })

            ctx$call_callbacks("on_train_end")

            if (!is.null(ctx$valid_data)) {
              ctx$data <- ctx$valid_data
              valid_loop(ctx, step)
            }

            ctx$call_callbacks("on_epoch_end")
          })
      }
    })

  ctx$call_callbacks("on_fit_end")
  ctx$clean()

  structure(
    ctx$state_dict(),
    class = "luz_module_fitted"
  )
}

#' Evaluates a fitted model on a dataset
#'
#' @param object A fitted model to evaluate.
#' @inheritParams fit.luz_module_generator
#' @param metrics A list of luz metrics to be tracked during evaluation. If `NULL`
#'   (default) then the same metrics that were used during training are tracked.
#'
#' @includeRmd man/rmd/evaluate.Rmd details
#'
#' @family training
#' @export
evaluate <- function(
    object,
    data,
    ...,
    metrics = NULL,
    callbacks = list(),
    accelerator = NULL,
    verbose = NULL,
    dataloader_options = NULL
) {

  enable_mps_fallback()

  # replace metrics for evaluate. metrics are attributes of luz modules.
  # after evaluation, metrics are replaced back for their original values.
  if (!is.null(metrics)) {
    model_metrics <- object$model$metrics
    on.exit({
      object$model$metrics <- model_metrics
    }, add = TRUE)
    object$model$metrics <- if (is_luz_metric_set(metrics)) {
      metrics
    } else {
      luz_metric_set(metrics)
    }
  }

  ctx <- evaluate_context$new(
    model = object$model,
    newdata = data,
    callbacks = callbacks,
    accelerator = accelerator,
    verbose = verbose,
    dataloader_options = dataloader_options,
    callbacks_default = default_evaluate_callbacks,
    opt_hparams = object$ctx$opt_hparams
  )

  on.exit({
    e <- rlang::current_env()
    rm(list = rlang::env_names(e), envir = e)
  }, add = TRUE)

  valid_loop(ctx, get_step(ctx))

  structure(
    ctx$state_dict(),
    class = "luz_module_evaluation"
  )
}

#' Create predictions for a fitted model
#'
#' @param object (fitted model) the fitted model object returned from [fit.luz_module_generator()]
#' @param newdata (dataloader, dataset, list or array) returning a list with at
#'   least 1 element. The other elements aren't used.
#' @inheritParams fit.luz_module_generator
#' @param ... Currently unused.
#'
#' @family training
#'
#' @importFrom stats predict
#' @export
predict.luz_module_fitted <- function(object, newdata, ..., callbacks = list(),
                                      accelerator = NULL, verbose = NULL,
                                      dataloader_options = NULL) {

  enable_mps_fallback()
  ctx <- predict_context$new(
    model = object$model,
    newdata = newdata,
    callbacks = callbacks,
    accelerator = accelerator,
    verbose = verbose,
    dataloader_options = dataloader_options,
    callbacks_default = default_predict_callbacks
  )

  pars <- rlang::list2(...)
  if (is.null(pars$stack))
    stack <- TRUE
  else
    stack <- pars$stack

  predict_fn <- if (is.null(ctx$model$predict)) ctx$model else ctx$model$predict

  on.exit({
    e <- rlang::current_env()
    rm(list = rlang::env_names(e), envir = e)
  }, add = TRUE)

  torch::with_no_grad({
    ctx$call_callbacks("on_predict_begin")
    with_handlers(
      !!! ctx$handlers,
      .expr = {
        coro::loop(for(batch in ctx$data) {
          ctx$batch <- batch
          ctx$call_callbacks("on_predict_batch_begin")
          ctx$pred[[length(ctx$pred) + 1]] <- do.call(predict_fn, list(ctx$input))
          ctx$call_callbacks("on_predict_batch_end")
        })
      }
    )
    ctx$call_callbacks("on_predict_end")
  })

  if (stack) {
    ctx$pred <- torch::torch_cat(ctx$pred)
  }

  ctx$pred
}

get_step <- function(ctx) {
  if (is.null(ctx$model$step))
    function() default_step(ctx)
  else
    ctx$model$step
}

valid_loop <- function(ctx, step) {
  torch::local_no_grad() # the whole validation loop has no grad enabled
  ctx$call_callbacks("on_valid_begin")

  ctx$iter <- 0L
  # helps making sure the dataloader workers are quickly deleted after the
  # evaluation loop
  local({
    next_batch <- as_iterator(ctx$data)
    while(!is_exhausted(batch <- next_batch())) {
      ctx$batch <- batch
      ctx$iter <- ctx$iter + 1L

      ctx$call_callbacks("on_valid_batch_begin")
      step()
      ctx$call_callbacks("on_valid_batch_end")
    }
  })

  ctx$call_callbacks("on_valid_end")

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

    ctx$loss_grad <- ctx$loss_fn(ctx$pred, ctx$target)
    ctx$loss[[ctx$opt_name]] <- ctx$loss_grad$detach()

    ctx$call_callbacks("on_train_batch_after_loss")

    ctx$call_callbacks("on_train_batch_before_backward")
    ctx$model$backward(ctx$loss_grad)

    ctx$call_callbacks("on_train_batch_before_step")
    ctx$step_opt(ctx$opt)
    ctx$opt$zero_grad()
    ctx$call_callbacks("on_train_batch_after_step")
  }
}

valid_one_batch <- function(ctx) {
  for (nm in names(ctx$optimizers)) {
    ctx$opt_name <- nm

    ctx$pred <- do.call(ctx$model, list(ctx$input))
    ctx$call_callbacks("on_valid_batch_after_pred")

    ctx$loss[[ctx$opt_name]] <- ctx$loss_fn(ctx$pred, ctx$target)
    ctx$call_callbacks("on_valid_batch_after_loss")
  }
}

initialize_callbacks <- function(callbacks, ctx) {
  cbs <- lapply(callbacks, function(cb) {
    assert_is_callback(cb)
    cb$set_ctx(ctx)
    bind_context(cb, ctx)
    cb
  })
  # reorder callbacks according to their weights
  weights <- sapply(cbs, function(x) x$weight %||% 0)
  cbs[order(weights)]
}

create_valid_data <- function(data, valid_data) {

  if (valid_data >= 1 || valid_data < 0)
    rlang::abort(sprintf("valid_data must be a value between 0 and 1, got %f", valid_data),
                 class = "value_error")

  if (torch::is_dataloader(data))
    rlang::abort(c("Can't create a validation set from the training dataloader.",
                   "Supply a torch dataset instead."), class = "value_error")

  data <- as_dataset(data)
  l <- length(data)

  id_valid <- sample.int(l, size = l*valid_data)
  id_train <- seq_len(length.out = l)[-id_valid]

  valid_data <- torch::dataset_subset(data, id_valid)
  data <- torch::dataset_subset(data, id_train)
  list(data, valid_data)
}

apply_dataloader_options <- function(data, valid_data, dataloader_options) {

  if (torch::is_dataloader(data) && !is.null(dataloader_options))
    rlang::abort("`dataloader_options` won't be used because `data` is already a dataloader.")

  if (torch::is_dataloader(valid_data) && !is.null(dataloader_options))
    rlang::warn("`dataloader_options` will be ignored for `valid_data` since it's already a dataloader")

  dataloader_options <- dataloader_options %||% list()

  if (is.null(dataloader_options$batch_size))
    dataloader_options$batch_size <- 32L

  if (!torch::is_dataloader(data)) {

    train_dl_options <- dataloader_options
    if (is.null(train_dl_options$shuffle))
      train_dl_options$shuffle <- TRUE

    # It's usually better to drop the last batch if its not the same size as the
    # other as it can have a large effect on results but based on very few obs.
    if (is.null(train_dl_options$drop_last))
      train_dl_options$drop_last <- TRUE

    data <- rlang::exec(as_dataloader, x = data, !!!train_dl_options)
  }

  if (!torch::is_dataloader(valid_data)) {
    valid_dl_options <- dataloader_options

    # probably on `predict`.
    if (is.null(data) && isTRUE(valid_dl_options$shuffle))
      rlang::warn("`shuffle=TRUE` will be ignored for predictions.")

    valid_dl_options$shuffle <- FALSE

    valid_data <- rlang::exec(as_dataloader, x = valid_data, !!!valid_dl_options)
  }

  list(data, valid_data)
}

#' Get metrics from the object
#' @param object The object to query for metrics.
#' @param ... Currently unused.
#' @returns A data.frame containing the metric values.
#' @export
get_metrics <- function(object, ...) {
  UseMethod("get_metrics")
}

#' @export
#' @describeIn get_metrics Extract metrics from a luz fitted model.
get_metrics.luz_module_fitted <- function(object, ...) {
  rlang::check_installed("dplyr")
  purrr::imap_dfr(object$records$metrics, make_metrics_df)
}

#' @export
get_metrics.luz_context <- get_metrics.luz_module_fitted

#' @export
get_metrics.luz_module_evaluation <- function(object, ...) {
  res <- get_metrics.luz_module_fitted(object)
  res[, c("metric", "value")]
}

can_use_mps <- function() {
  arch <- Sys.info()["machine"]
  identical(Sys.getenv("LUZ_DISABLE_MPS", "FALSE"), "FALSE") && "arm64" %in% arch && torch::backends_mps_is_available()
}

enable_mps_fallback <- function() {
  if (!can_use_mps())
    return(invisible(NULL))

  fallback <- Sys.getenv("PYTORCH_ENABLE_MPS_FALLBACK", unset = "")
  if (fallback == "") {
    if (!identical(Sys.getenv("TESTTHAT"), "true")) {
      cli::cli_warn(c(
        paste0(
          "Some torch operators might not yet be implemented for the MPS device. ",
          "A temporary fix is to set the {.var PYTORCH_ENABLE_MPS_FALLBACK=1} to ",
          "use the CPU as a fall back for those operators:"),
        i = paste0(
          "Add {.var PYTORCH_ENABLE_MPS_FALLBACK=1} to your {.var .Renviron} file, ",
          "for example use {.fn usethis::edit_r_environ}."),
        x = paste0(
          "Using {.var Sys.setenv()} doesn't work because the env var must be ",
          "set before R starts.")
      ))
    }
  }
  invisible(NULL)
}
