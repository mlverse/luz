test_that("callback profile", {

  model <- get_model()
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam,
    )

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, verbose = FALSE, epochs = 5, valid_data = dl)

  expect_length(output$records$profile$fit, 1)
  expect_length(output$records$profile$epoch, 5)
  expect_length(output$records$profile$train, 5)
  expect_length(output$records$profile$valid, 5)

})
