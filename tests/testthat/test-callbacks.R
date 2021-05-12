test_that("early stopping", {
  torch::torch_manual_seed(1)
  set.seed(1)

  model <- get_model()
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam,
    )

  expect_snapshot({
    expect_message({
      output <- mod %>%
        set_hparams(input_size = 10, output_size = 1) %>%
        fit(dl, verbose = TRUE, epochs = 25, callbacks = list(
          luz_callback_early_stopping(monitor = "train_loss", patience = 1)
        ))
    })
  })

  expect_snapshot({
    expect_message({
      output <- mod %>%
        set_hparams(input_size = 10, output_size = 1) %>%
        fit(dl, verbose = TRUE, epochs = 25, callbacks = list(
          luz_callback_early_stopping(monitor = "train_loss", patience = 5,
                                      baseline = 0.001)
        ))
    })
  })

  x <- 0
  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, verbose = FALSE, epochs = 25, callbacks = list(
      luz_callback_early_stopping(monitor = "train_loss", patience = 5,
                                  baseline = 0.001),
      luz_callback(on_early_stopping = function() {
        x <<- 1
      })()
    ))

  expect_equal(x, 1)
})

