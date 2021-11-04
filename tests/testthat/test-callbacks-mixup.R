test_that("mixup logic works", {

  dl <- get_categorical_dl(x_size = 768)
  first <- coro::collect(dl, 1)[[1]]

  c(mixed_x, stacked_y_with_weights) %<-% nnf_mixup(
    first$x,
    first$y,
    torch::torch_tensor(1:10),
    torch::torch_tensor(rep(0.9, 10))$view(c(10, 1)))

  expect_equal_to_tensor(mixed_x[1, ] %>% torch::torch_mean(), first$x[1, ] %>% torch::torch_mean())
  expect_equal_to_tensor(stacked_y_with_weights[[1]][[1]], stacked_y_with_weights[[1]][[2]])

  c(mixed_x, stacked_y_with_weights) %<-% mixup(
    first$x,
    first$y,
    torch::torch_tensor(10:1),
    torch::torch_tensor(rep(0.9, 10))$view(c(10, 1)))

  expect_not_equal_to_tensor(mixed_x[1, ] %>% torch::torch_mean(), first$x[1, ] %>% torch::torch_mean())
  expect_not_equal_to_tensor(stacked_y_with_weights[[1]][[1]], stacked_y_with_weights[[1]][[2]])

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

