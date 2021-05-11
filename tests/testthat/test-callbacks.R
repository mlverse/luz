test_that("early stopping", {
  model <- get_model()
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = optim_adam,
    )

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, verbose = TRUE, callbacks = list(
      luz_callback_early_stopping(monitor = "train_loss", patience = 1)
    ))

})
