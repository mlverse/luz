test_that("callback lr scheduler", {

  skip_on_os("windows")

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
        fit(dl, verbose = FALSE, epochs = 5, callbacks = list(
          luz_callback_lr_scheduler(torch::lr_multiplicative, verbose = TRUE,
                                    lr_lambda = function(epoch) 0.5)
        ))
    })
  })

})

test_that("csv callback", {

  model <- get_model()
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam,
    )

  tmp <- tempfile()

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, verbose = FALSE, epochs = 5, callbacks = list(
      luz_callback_csv_logger(tmp)
    ))

  x <- read.table(tmp, header = TRUE, sep = ",")
  expect_equal(nrow(x), 5)
  expect_equal(names(x), c("epoch", "set", "loss"))

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, verbose = FALSE, epochs = 5, valid_data = dl, callbacks = list(
      luz_callback_csv_logger(tmp)
    ))

  x <- read.table(tmp, header = TRUE, sep = ",")

  expect_equal(nrow(x), 10)
  expect_equal(names(x), c("epoch", "set", "loss"))

})

test_that("progressbar appears with training and validation", {

  torch::torch_manual_seed(1)
  set.seed(1)

  model <- get_model()
  dl <- get_test_dl(len = 500)

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam,
    )

  withr::with_options(list(luz.force_progress_bar = TRUE,
                           luz.show_progress_bar_eta = FALSE,
                           width = 80), {
    expect_snapshot({
      expect_message({
        output <- mod %>%
          set_hparams(input_size = 10, output_size = 1) %>%
          fit(dl, verbose = TRUE, epochs = 2, valid_data = dl)
      })
    })
  })

})
