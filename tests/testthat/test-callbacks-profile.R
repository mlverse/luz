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

  expect_length(output$ctx$records$profile$fit, 1)
  expect_length(output$ctx$records$profile$epoch, 5)

  expect_length(output$ctx$records$profile$train, 5)
  expect_length(output$ctx$records$profile$train_step, 50)
  expect_length(output$ctx$records$profile$train_batch, 50)

  expect_length(output$ctx$records$profile$valid, 5)
  expect_length(output$ctx$records$profile$valid_step, 50)
  expect_length(output$ctx$records$profile$valid_batch, 50)

})
