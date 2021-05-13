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

  # the new callback breakpoint is used
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

  # metric that is not the loss

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam,
      metrics = luz_metric_mae()
    )

  expect_snapshot({
    expect_message({
      output <- mod %>%
        set_hparams(input_size = 10, output_size = 1) %>%
        fit(dl, verbose = TRUE, epochs = 25, callbacks = list(
          luz_callback_early_stopping(monitor = "train_mae", patience = 2,
                                      baseline = 0.91)
        ))
    })
  })


})

test_that("model checkpoint callback works", {


  torch::torch_manual_seed(1)
  set.seed(1)

  model <- get_model()
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam,
    )

  tmp <- tempfile(fileext = "/")

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, verbose = FALSE, epochs = 5, callbacks = list(
      luz_callback_model_checkpoint(path = tmp, monitor = "train_loss",
                                    save_best_only = FALSE)
    ))

  files <- fs::dir_ls(tmp)
  expect_length(files, 5)

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, verbose = FALSE, epochs = 10, callbacks = list(
      luz_callback_model_checkpoint(path = tmp, monitor = "train_loss",
                                    save_best_only = TRUE)
    ))

  files <- fs::dir_ls(tmp)
  expect_length(files, 5)

})
