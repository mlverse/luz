test_that("End to end fitting", {

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
