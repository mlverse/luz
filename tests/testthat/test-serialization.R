test_that("serialization works as expected", {

  torch::torch_manual_seed(1)
  set.seed(1)

  model <- get_model()
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam,
    )

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, verbose = FALSE, epochs = 1)

  path <- tempfile(pattern="rds")
  luz_save(output, path)

  test <- get_test_dl()

  expected_predictions <- predict(output, test)
  params <- output$model$state_dict()
  rm(output)
  gc()

  output <- luz_load(path)

  params2 <- output$model$state_dict()
  actual_predictions <- predict(output, test)

  expect_equal(
    as.array(actual_predictions$to(device="cpu")),
    as.array(expected_predictions$to(device="cpu"))
  )
})

test_that("serialization works when model uses `ctx` in forward", {

  model <- torch::nn_module(
    initialize = function() {
      self$fc <- torch::nn_linear(10, 1, bias = FALSE)
    },
    forward = function(x) {
      if (ctx$training)
        self$fc(torch::torch_ones_like(x))
      else
        self$fc(torch::torch_zeros_like(x))
    }
  )

  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam,
    )

  output <- mod %>%
    set_hparams() %>%
    fit(dl, verbose = FALSE, epochs = 1)


  pred <- predict(output, dl)
  expect_equal(as.array(pred), as.array(torch::torch_zeros(100,1)))

  output$ctx$training <- TRUE

  path <- tempfile(pattern="rds")
  luz_save(output, path)

  rm(output)
  gc()

  output <- luz_load(path)
  pred <- predict(output, dl)
  expect_equal(
    as.array(pred$to(device="cpu")),
    as.array(torch::torch_zeros(100,1))
  )
})
