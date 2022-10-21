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

test_that("gradient clip works correctly", {

  model <- get_model()
  dl <- get_test_dl(len = 500)

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam,
    )

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, verbose = FALSE, epochs = 2, valid_data = dl,
        callbacks = list(luz_callback_gradient_clip(max_norm = 0)))

  # we expect that no learning happened thus the loss is identicall
  # acrosss all metrics.
  expect_length(unique(get_metrics(output)$value), 1)
  expect_length(get_metrics(output)$value, 4)

  expect_error(luz_callback_gradient_clip(max_norm = "a"), "max_norm")
  expect_error(luz_callback_gradient_clip(norm_type = "a"), "norm_type")
})

test_that("improve error message when you provide a unitinitilized callback", {

  skip_on_os("windows")

  x <- torch_randn(1000, 10)
  y <- torch_randn(1000, 1)

  model <- nn_linear %>%
    setup(optimizer = optim_sgd, loss = nnf_mse_loss) %>%
    set_hparams(in_features = 10, out_features = 1) %>%
    set_opt_hparams(lr = 0.01)

  expect_snapshot_error({
    model %>% fit(list(x, y), callbacks = list(luz_callback_auto_resume))
  })

})
