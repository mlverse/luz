test_that("Fully managed", {

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
    fit(dl, valid_data = dl, verbose = FALSE)

  expect_s3_class(output, "nn_module")
})

test_that("Custom optimizer", {

  model <- get_model()
  model <- torch::nn_module(
    inherit = model,
    optimizer = function() {
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

  expect_s3_class(output, "nn_module")
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
    optimizer = function() {
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

  expect_s3_class(output, "nn_module")
})
