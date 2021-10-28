test_that("mixup works for 1d input", {

  dl <- get_categorical_dl(x_size = 768)

  model <- get_model()
  mod <- model %>%
    setup(
      loss = function(input, target) luz_mixup_cross_entropy(input = input, target = target, ignore_index = 222),
      optimizer = torch::optim_adam,
    ) %>%
    set_hparams(input_size = 768, output_size = 10) %>%
    fit(dl, verbose = TRUE, epochs = 2, valid_data = dl,
        callbacks = list(luz_callback_mixup()))


})

test_that("mixup works for 2d input", {

  dl <- get_categorical_dl(x_size = c(28, 28), num_classes = 3)

  model <- get_model()
  mod <- model %>%
    setup(
      loss = function(input, target) luz_mixup_cross_entropy(input = input, target = target, ignore_index = 222),
      optimizer = torch::optim_adam,
    ) %>%
    set_hparams(input_size = c(28, 28), output_size = 3) %>%
    fit(dl, verbose = TRUE, epochs = 2, valid_data = dl,
        callbacks = list(luz_callback_mixup()))


})

test_that("mixup works for 3d input", {

  dl <- get_categorical_dl(x_size = c(3, 28, 28), num_classes = 33)

  model <- get_model()
  mod <- model %>%
    setup(
      loss = function(input, target) luz_mixup_cross_entropy(input = input, target = target, ignore_index = 222),
      optimizer = torch::optim_adam,
    ) %>%
    set_hparams(input_size = c(3, 28, 28), output_size = 33) %>%
    fit(dl, verbose = TRUE, epochs = 2, valid_data = dl,
        callbacks = list(luz_callback_mixup()))


})

