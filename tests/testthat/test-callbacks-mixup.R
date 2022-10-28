test_that("mixup logic works", {

  x <- torch::torch_ones(c(10, 768))
  y <- torch::torch_ones(10)

  c(mixed_x, stacked_y_with_weights) %<-% nnf_mixup(
    x,
    y,
    torch::torch_tensor(rep(0.9, 10))$view(c(10, 1)))

  expect_equal_to_tensor(mixed_x[1, ] %>% torch::torch_mean(), x[1, ] %>% torch::torch_mean())
  expect_equal_to_tensor(stacked_y_with_weights[[1]][[1]], stacked_y_with_weights[[1]][[2]])

})

test_that("mixup callback successful for 1d input", {

  dl <- get_categorical_dl(x_size = 768)

  model <- get_model()
  expect_silent({
  mod <- model %>%
    setup(
      loss = nn_mixup_loss(torch::nn_cross_entropy_loss(ignore_index = 222)),
      optimizer = torch::optim_adam,
    ) %>%
    set_hparams(input_size = 768, output_size = 10) %>%
    fit(dl, verbose = FALSE, epochs = 2, valid_data = dl,
        callbacks = list(luz_callback_mixup()))
  })
})

test_that("mixup works for 2d input", {

  dl <- get_categorical_dl(x_size = c(28, 28), num_classes = 3)

  model <- get_model()

  expect_silent({
  mod <- model %>%
    setup(
      loss = nn_mixup_loss(torch::nn_cross_entropy_loss()),
      optimizer = torch::optim_adam,
    ) %>%
    set_hparams(input_size = c(28, 28), output_size = 3) %>%
    fit(dl, verbose = FALSE, epochs = 2, valid_data = dl,
        callbacks = list(luz_callback_mixup()))
  })
})

test_that("mixup works for 3d input", {

  dl <- get_categorical_dl(x_size = c(3, 28, 28), num_classes = 33)

  model <- get_model()

  expect_silent({
    mod <- model %>%
      setup(
        loss = nn_mixup_loss(torch::nn_cross_entropy_loss()),
        optimizer = torch::optim_adam,
      ) %>%
      set_hparams(input_size = c(3, 28, 28), output_size = 33) %>%
      fit(dl, verbose = FALSE, epochs = 2, valid_data = dl,
          callbacks = list(luz_callback_mixup()))
  })
})

test_that("can use mixup with accuracy", {
  # tests if it's possible to use mixup and in the same time compute accuracy
  # for the validation set.

  x <- torch_randn(1000, 10)
  y <- torch_randint(1, 2, size = 1000, dtype = torch_int64())

  model <- nn_linear %>%
    setup(
      loss = nn_cross_entropy_loss(reduction = "none"),
      optimizer = optim_sgd,
      metrics = luz_metric_set(
        valid_metrics = luz_metric_accuracy()
      )
    ) %>%
    set_hparams(in_features = 10, out_features = 2) %>%
    set_opt_hparams(lr = 0.001)

  expect_error(
    result <- model %>% fit(
      list(x, y),
      valid_data = 0.2,
      callbacks = list(
        luz_callback_mixup(auto_loss = TRUE)
      ),
      verbose = FALSE
    ),
    regexp = NA
  )

  expect_true("acc" %in% get_metrics(result)$metric)
})

