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
    fit(dl, valid_data = dl)

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
    fit(dl, valid_data = dl)

  expect_s3_class(output, "nn_module")
})
