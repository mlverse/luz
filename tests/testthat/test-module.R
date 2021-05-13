test_that("Fully managed", {

  model <- get_model()
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam
    )

  expect_s3_class(mod, "luz_module_generator")

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, valid_data = dl, verbose = FALSE)

  expect_s3_class(output, "luz_module_fitted")
})

test_that("Custom optimizer", {

  model <- get_model()
  model <- torch::nn_module(
    inherit = model,
    set_optimizers = function() {
      optim_adam(self$parameters, lr = 0.01)
    }
  )
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
    )

  expect_s3_class(mod, "luz_module_generator")

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, valid_data = dl, verbose = FALSE)

  expect_s3_class(output, "luz_module_fitted")
})

test_that("Multiple optimizers", {

  model <- get_model()
  module <- torch::nn_module(
    initialize = function(input_size = 10, output_size = 1) {
      self$model1 = model(input_size, output_size)
      self$model2 <- model(input_size, output_size)
    },
    forward = function(x) {
      self$model1(x) + self$model2(x)
    },
    set_optimizers = function() {
      list(
        one = torch::optim_adam(self$model1$parameters, lr = 0.01),
        two = torch::optim_adam(self$model2$parameters, lr = 0.01)
      )
    }
  )
  dl <- get_dl()

  mod <- module %>%
    setup(
      loss = torch::nn_mse_loss(),
    )

  expect_s3_class(mod, "luz_module_generator")

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, valid_data = dl, verbose = FALSE)

  expect_s3_class(output, "luz_module_fitted")
})

test_that("can train without a validation dataset", {

  model <- get_model()
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = optim_adam
    )

  expect_s3_class(mod, "luz_module_generator")

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, verbose = FALSE)

  expect_s3_class(output, "luz_module_fitted")
})

test_that("predict works for modules", {

  model <- get_model()
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = optim_adam
    )

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, verbose = FALSE)


  pred <- predict(output, dl)
  pred2 <- predict(output, dl)

  expect_equal(pred$shape, c(100, 1))
  expect_equal(as.array(pred$to(device = "cpu")), as.array(pred2$to(device="cpu")))

  # try with a different dataloader
  dl <- get_dl()
  pred <- predict(output, dl)
  pred2 <- predict(output, dl)
  expect_equal(as.array(pred$to(device = "cpu")), as.array(pred2$to(device="cpu")))

})
