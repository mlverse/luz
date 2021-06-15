test_that("print module generator", {

  model <- get_model()
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam
    )

  expect_snapshot(
    print(mod)
  )

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    set_opt_hparams(lr = 0.001) %>%
    fit(dl, epochs = 1, verbose = FALSE, valid_data = dl)

  expect_output(
    print(output),
    regexp = "Total time:"
  )

})
