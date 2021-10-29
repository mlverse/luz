test_that("mixup works for 1d input", {

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

