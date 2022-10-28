test_that("Can create a simple metric", {

  metric <- luz_metric(
    name = "MyMetric",
    initialize = function(y) {
      self$z <- 0
      self$y <- y
    },
    update = function(preds, target) {
      self$z <- self$z + self$y
    },
    compute = function() {
      self$z
    }
  )

  # this creates an inherited class with partialized arguments.
  init <- metric(y = 1)
  expect_s3_class(init, "R6ClassGenerator")

  # initializes an instance of the class
  m <- init$new()
  expect_s3_class(m, "MyMetric")

  # run update step
  m$update(1,1)

  expect_equal(m$compute(), 1)
})

test_that("mae works", {

  x <- torch::torch_randn(100, 100, device = get_device())
  y <- torch::torch_randn(100, 100, device = get_device())

  m <- luz_metric_mae()
  m <- m$new()$to(device = get_device())

  m$update(x, y)
  o <- m$compute()
  eo <- mean(abs(as.array(x$cpu()) - as.array(y$cpu())))

  expect_equal(o, eo, tolerance = 1e-5)
})

test_that("mse works", {

  x <- torch::torch_randn(100, 100, device = get_device())
  y <- torch::torch_randn(100, 100, device = get_device())

  m <- luz_metric_mse()
  m <- m$new()$to(device = get_device())

  m$update(x, y)
  o <- m$compute()
  eo <- mean((as.array(x$cpu()) - as.array(y$cpu()))^2)

  expect_equal(o, eo, tolerance = 1e-5)
})

test_that("rmse works", {

  x <- torch::torch_randn(100, 100, device = get_device())
  y <- torch::torch_randn(100, 100, device = get_device())

  m <- luz_metric_rmse()
  m <- m$new()$to(device = get_device())

  m$update(x, y)
  o <- m$compute()
  eo <- sqrt(mean((as.array(x$cpu()) - as.array(y$cpu()))^2))

  expect_equal(o, eo, tolerance = 1e-5)
})

test_that("binary accuracy with logits", {

  m <- luz_metric_binary_accuracy_with_logits(threshold = 0.5)
  m <- m$new()$to(device = get_device())

  x <- torch_randn(100, device = get_device())
  y <- torch_randint(0, 1, 100, device = get_device())

  m$update(x, y)
  expect_equal(
    m$compute(),
    mean(as.array((x > 0)$cpu()) == as.array(y$cpu()))
  )

})

test_that("metrics works within models", {

  dl <- get_binary_dl()
  model <- torch::nn_linear

  mod <- model %>%
    setup(
      loss = torch::nn_bce_with_logits_loss(),
      optimizer = torch::optim_adam,
      metrics = list(
        luz_metric_binary_accuracy_with_logits(),
        luz_metric_binary_auroc(from_logits = TRUE),
        luz_metric_binary_accuracy()
      )
    )

  expect_error(
    output <- mod %>%
      set_hparams(in_features = 10, out_features = 1) %>%
      fit(dl, epochs = 1, verbose = FALSE),
    regexp = NA
  )

  expect_length(
    output$records$metrics$train[[1]],
    4
  )

})

test_that("can specify metrics for training and validation", {

  x <- torch_randn(1000, 10)
  y <- torch_randn(1000, 1)

  model <- nn_linear %>%
    setup(optimizer = optim_sgd, loss = torch::nn_mse_loss(),
          metrics = luz_metric_set(
            metrics = c(luz_metric_mae()),
            valid_metrics = c(luz_metric_rmse()),
            train_metrics = c(luz_metric_mse())
          )) %>%
    set_hparams(in_features = 10, out_features = 1) %>%
    set_opt_hparams(lr = 0.001)


  res <- model %>%
    fit(list(x, y), epochs = 5, valid_data = list(x, y), verbose = FALSE)

  metrics <- get_metrics(res)
  expect_equal(
    unique(metrics[metrics$set== "valid", "metric"]),
    c("loss", "mae", "rmse")
  )
  expect_equal(
    unique(metrics[metrics$set== "train", "metric"]),
    c("loss", "mae", "mse")
  )
  expect_error(regexp = NA, {
    plot(res)
  })
})
