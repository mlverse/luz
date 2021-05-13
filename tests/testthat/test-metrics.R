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

  x <- torch::torch_randn(100, 100)
  y <- torch::torch_randn(100, 100)

  m <- luz_metric_mae()
  m <- m$new()

  m$update(x, y)
  o <- m$compute()
  eo <- mean(abs(as.array(x) - as.array(y)))

  expect_equal(o, eo, tolerance = 1e-5)
})

test_that("mse works", {

  x <- torch::torch_randn(100, 100)
  y <- torch::torch_randn(100, 100)

  m <- luz_metric_mse()
  m <- m$new()

  m$update(x, y)
  o <- m$compute()
  eo <- mean((as.array(x) - as.array(y))^2)

  expect_equal(o, eo, tolerance = 1e-5)
})

test_that("rmse works", {

  x <- torch::torch_randn(100, 100)
  y <- torch::torch_randn(100, 100)

  m <- luz_metric_rmse()
  m <- m$new()

  m$update(x, y)
  o <- m$compute()
  eo <- sqrt(mean((as.array(x) - as.array(y))^2))

  expect_equal(o, eo, tolerance = 1e-5)
})

test_that("binary accuracy with logits", {

  m <- luz_metric_binary_accuracy_with_logits(threshold = 0.5)
  m <- m$new()

  x <- torch_randn(100)
  y <- torch_randint(0, 1, 100)

  m$update(x, y)
  expect_equal(
    m$compute(),
    mean(as.array(x > 0) == as.array(y))
  )

})
