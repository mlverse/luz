test_that("Fully managed", {

  model <- get_model()
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam
    )

  expect_s3_class(mod, "luz_module_generator")

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, valid_data = dl, verbose = FALSE)

  expect_s3_class(output, "luz_module_fitted")
})

test_that("Custom optimizer", {

  model <- get_model()
  model <- torch::nn_module(
    inherit = model,
    set_optimizers = function() {
      torch::optim_adam(self$parameters, lr = 0.01)
    }
  )
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
    )

  expect_s3_class(mod, "luz_module_generator")

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, valid_data = dl, verbose = FALSE)

  expect_s3_class(output, "luz_module_fitted")
})

test_that("Multiple optimizers", {

  model <- get_model()
  module <- torch::nn_module(
    initialize = function(input_size = 10, output_size = 1) {
      self$model1 = model(input_size, output_size)
      self$model2 <- model(input_size, output_size)
    },
    forward = function(x) {
      self$model1(x) + self$model2(x)
    },
    set_optimizers = function() {
      list(
        one = torch::optim_adam(self$model1$parameters, lr = 0.01),
        two = torch::optim_adam(self$model2$parameters, lr = 0.01)
      )
    }
  )
  dl <- get_dl()

  mod <- module %>%
    setup(
      loss = torch::nn_mse_loss(),
    )

  expect_s3_class(mod, "luz_module_generator")

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, valid_data = dl, verbose = FALSE)

  expect_s3_class(output, "luz_module_fitted")
})

test_that("can train without a validation dataset", {

  model <- get_model()
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam
    )

  expect_s3_class(mod, "luz_module_generator")

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, verbose = FALSE)

  expect_s3_class(output, "luz_module_fitted")
})

test_that("predict works for modules", {

  model <- get_model()
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam
    )

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, verbose = FALSE)


  pred <- predict(output, dl)
  pred2 <- predict(output, dl)

  expect_equal(pred$shape, c(100, 1))
  expect_equal(as.array(pred$to(device = "cpu")), as.array(pred2$to(device="cpu")))

  # try with a different dataloader
  dl <- get_dl()
  pred <- predict(output, dl)
  pred2 <- predict(output, dl)
  expect_equal(as.array(pred$to(device = "cpu")), as.array(pred2$to(device="cpu")))

})

test_that("predict can use a progress bar", {

  model <- get_model()
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam
    )

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    set_opt_hparams(lr = 0.001) %>%
    fit(dl, epochs = 1, verbose = FALSE)

  dl <- get_dl(len = 500)

  withr::with_options(
    list(luz.force_progress_bar = TRUE,
         luz.show_progress_bar_eta = FALSE,
         width = 80),
    {
      expect_snapshot(
        pred <- predict(output, dl, verbose=TRUE)
      )
    }
  )

  expect_equal(output$ctx$hparams$input_size, 10)
  expect_equal(output$ctx$opt_hparams$lr, 0.001)
})

test_that("valid_data works", {

  model <- get_model()
  model <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam
    ) %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    set_opt_hparams(lr = 0.001)

  fitted <- model %>% fit(
    list(torch::torch_randn(100,10), torch::torch_randn(100, 1)),
    epochs = 10,
    valid_data = 0.1,
    verbose = FALSE
  )

  expect_true("valid" %in% get_metrics(fitted)$set)

  expect_error(class= "value_error", regexp = "2", {
    model %>% fit(
      list(torch::torch_randn(100,10), torch::torch_randn(100, 1)),
      epochs = 10,
      valid_data = 2,
      verbose = FALSE
    )
  })

  expect_error(class= "value_error", regexp = "-1", {
    model %>% fit(
      list(torch::torch_randn(100,10), torch::torch_randn(100, 1)),
      epochs = 10,
      valid_data = -1,
      verbose = FALSE
    )
  })

  dl <- get_dl()
  expect_error(class= "value_error", regexp = "dataloader", {
    model %>% fit(
      dl,
      epochs = 10,
      valid_data = 0.2,
      verbose = FALSE
    )
  })

})

test_that("we can pass dataloader_options", {

  model <- get_model()
  model <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam
    ) %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    set_opt_hparams(lr = 0.001)

  x <- list(torch::torch_randn(100,10), torch::torch_randn(100, 1))

  fitted <- model %>% fit(
    x,
    epochs = 1,
    valid_data = 0.1,
    verbose = FALSE,
    dataloader_options = list(batch_size = 2, shuffle = FALSE)
  )

  expect_length(fitted$records$profile$train_step, 45)

  dl <- get_dl()
  expect_error(regexp = "already a dataloader", {
    model %>% fit(
      dl,
      epochs = 1,
      verbose = FALSE,
      dataloader_options = list(batch_size = 2, shuffle = FALSE)
    )
  })

  expect_warning(regexp = "already a dataloader", {
    model %>% fit(
      x,
      epochs = 1,
      verbose = FALSE,
      valid_data = dl,
      dataloader_options = list(batch_size = 2, shuffle = FALSE)
    )
  })

  pred <- predict(fitted, x, dataloader_options = list(batch_size = 45, drop_last = TRUE))
  expect_tensor_shape(pred, c(90, 1))

  expect_warning(regexp = "already a dataloader", {
    predict(fitted, dl, dataloader_options = list(batch_size = 45, drop_last = TRUE))
  })

  expect_warning(regexp = "ignored for predictions", {
    predict(fitted, x, dataloader_options = list(shuffle = TRUE))
  })

})

test_that("evaluate works", {

  set.seed(1)
  torch_manual_seed(1)

  model <- get_model()
  model <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam,
      metrics = list(
        luz_metric_mae(),
        luz_metric_mse(),
        luz_metric_rmse()
      )
    ) %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    set_opt_hparams(lr = 0.001)

  x <- list(torch::torch_randn(100,10), torch::torch_randn(100, 1))

  fitted <- model %>% fit(
    x,
    epochs = 1,
    verbose = FALSE,
    dataloader_options = list(batch_size = 2, shuffle = FALSE)
  )

  e <- evaluate(fitted, x)

  expect_equal(nrow(get_metrics(e)), 4)
  expect_equal(ncol(get_metrics(e)), 2)

  expect_snapshot(print(e))
})

test_that("cutom backward", {

  model <- get_model()
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam,
      backward = function(x) {
        x$backward()
        if (ctx$iter == 1 && ctx$epoch == 1) {
          print("hello")
        }
      }
    )

  expect_s3_class(mod, "luz_module_generator")

  expect_output(regexp = "hello", {
    output <- mod %>%
      set_hparams(input_size = 10, output_size = 1) %>%
      fit(dl, valid_data = dl, verbose = FALSE)
  })
  expect_s3_class(output, "luz_module_fitted")
})

test_that("luz module has a device arg", {

  mod <- get_model() %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam
    )

  modul <- mod(1,1)

  expect_true(
    modul$device == torch_device("cpu")
  )

  mod <- nn_module(
    initialize = function() {
      self$par <- torch::nn_parameter(torch_randn(10, 10))
    },
    forward = function(x) {
      self$par
    },
    active = list(
      device = function() {
        "hello"
      }
    )
  )

  model <- mod %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam
    )

  modul <- mod()
  expect_equal(modul$device, "hello")

})
