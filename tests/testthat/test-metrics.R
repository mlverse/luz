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
