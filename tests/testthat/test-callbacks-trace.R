test_that("trace callback for minimum model", {

  model <- torch::nn_linear
  model <- setup(model,
                 loss = torch::nn_mse_loss(),
                 optimizer = torch::optim_adam)
  model <- set_hparams(model, in_features = 10, out_features = 10)


  ds <- torch::tensor_dataset(x = torch::torch_randn(10, 10), y = torch::torch_randn(10, 10))
  dl <- torch::dataloader(ds, batch_size = 2)

  fitted <- fit(model, dl, epochs = 5, callbacks = list(luz_callback_trace()), verbose = FALSE)

  expect_s3_class(fitted, "luz_module_fitted")
})

test_that("works by disabling the checks", {
  # this model is non-deterministic for the training loop because it includes
  # a dropout, so it should if check_train = TRUE

  model <- torch::nn_module(
    initialize = function() {
      self$linear <- torch::nn_linear(10, 10)
      self$dropout <- torch::nn_dropout()
    },
    forward = function(x) {
      x %>% self$linear() %>% self$dropout()
    }
  )

  model <- setup(model,
                 loss = torch::nn_mse_loss(),
                 optimizer = torch::optim_adam)


  ds <- torch::tensor_dataset(x = torch::torch_randn(10, 10), y = torch::torch_randn(10, 10))
  dl <- torch::dataloader(ds, batch_size = 2)

  expect_error(
    fitted <- fit(model, dl, epochs = 5, valid_data = dl, callbacks = list(luz_callback_trace()), verbose = FALSE),
    regexp = "Traced model didn't"
  )

  fitted <- fit(model, dl, epochs = 5, valid_data = dl, callbacks = list(luz_callback_trace(check_train = FALSE)), verbose = FALSE)
  expect_s3_class(fitted, "luz_module_fitted")
})

test_that("works by disabling validation checks too", {

  model <- torch::nn_module(
    initialize = function() {
      self$linear <- torch::nn_linear(10, 10)
    },
    forward = function(x) {
      self$linear(x) + torch::torch_randn(1)
    }
  )

  model <- setup(model,
                 loss = torch::nn_mse_loss(),
                 optimizer = torch::optim_adam)


  ds <- torch::tensor_dataset(x = torch::torch_randn(10, 10), y = torch::torch_randn(10, 10))
  dl <- torch::dataloader(ds, batch_size = 2)

  expect_error(
    fitted <- fit(model, dl, epochs = 5, valid_data = dl, callbacks = list(luz_callback_trace()), verbose = FALSE),
    regexp = "Traced model didn't"
  )

  expect_error(
    fitted <- fit(model, dl, epochs = 5, valid_data = dl, callbacks = list(luz_callback_trace(check_train = FALSE)), verbose = FALSE),
    regexp = "Traced model didn't"
  )

  fitted <- fit(model, dl, epochs = 5, valid_data = dl, callbacks = list(luz_callback_trace(check_train = FALSE, check_valid = FALSE)), verbose = FALSE)
  expect_s3_class(fitted, "luz_module_fitted")
})

test_that("parameters are correctly updated", {

  model <- torch::nn_linear
  model <- setup(model,
                 loss = torch::nn_mse_loss(),
                 optimizer = torch::optim_sgd)
  model <- set_hparams(model, in_features = 2, out_features = 1)
  model <- set_opt_hparams(model, lr = 0.1)


  x <- torch::torch_randn(10, 2)
  y <- torch::torch_sum(x, 2, keepdim = TRUE)

  ds <- torch::tensor_dataset(x = x, y = y)
  dl <- torch::dataloader(ds, batch_size = 5)

  fitted <- fit(model, dl, epochs = 20, verbose = FALSE, callbacks = luz_callback_trace())
  expect_lt(tail(fitted$ctx$get_metrics_df()$value, 1), 0.1)

})
