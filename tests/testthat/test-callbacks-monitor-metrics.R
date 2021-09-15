test_that("early stopping with patience = 1", {

  fit_with_callback <- function(cb, epochs = 25) {
    model <- get_model()
    dl <- get_dl()

    suppressMessages({
      expect_message({
        model %>%
          setup(
            loss = torch::nn_mse_loss(),
            optimizer = torch::optim_adam,
          ) %>%
          set_hparams(input_size = 10, output_size = 1) %>%
          fit(dl, verbose = TRUE, epochs = epochs, callbacks = list(cb))
      })
    })
  }

  # since min_delta = 100 (large number) we expect that we will only train for
  # 2 epochs. The first one being to get a 'current best' value and the second
  # one will show no improvement thus stop training.
  mod <- fit_with_callback(luz_callback_early_stopping(
    monitor = "train_loss",
    patience = 1,
    min_delta = 100
  ))
  expect_equal(nrow(get_metrics(mod)), 2)

  # when patience equal 2 we expect to train for at least 2 epochs.
  mod <- fit_with_callback(luz_callback_early_stopping(
    monitor = "train_loss",
    patience = 2,
    min_delta = 100
  ))
  expect_equal(nrow(get_metrics(mod)), 3)

  # we have now scpecified that min_epochs = 5, so we must traiin for at least 5
  # epochs. However, when we are done the counter should be already updated and
  # ready to stop training.
  mod <- fit_with_callback(epochs = c(5, 25), luz_callback_early_stopping(
    monitor = "train_loss",
    patience = 2,
    min_delta = 100
  ))
  expect_equal(nrow(get_metrics(mod)), 5)

  # if the baseline is 0, we expect to stop in the first epoch.
  mod <- fit_with_callback(epochs = c(1, 25), luz_callback_early_stopping(
    monitor = "train_loss",
    patience = 1,
    baseline = 0
  ))
  expect_equal(nrow(get_metrics(mod)), 1)

})


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
          luz_callback_early_stopping(monitor = "train_loss", patience = 1,
                                      min_delta = 0.02)
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
                                      baseline = 0.91, min_delta = 0.01)
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

  tmp <- tempfile(fileext = "/")

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, verbose = FALSE, epochs = 10, callbacks = list(
      luz_callback_model_checkpoint(path = tmp, monitor = "train_loss",
                                    save_best_only = TRUE)
    ))

  files <- fs::dir_ls(tmp)
  expect_length(files, 10)

  torch::torch_manual_seed(2)
  set.seed(2)

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
                                    save_best_only = TRUE)
    ))

  files <- fs::dir_ls(tmp)
  expect_length(files, 5)

})
