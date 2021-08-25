test_that("as_dataloader fails", {
  t <- structure(1, class = c("test", "hi"))
  expect_error(
    as_dataloader(t),
    class = "value_error"
  )
})

test_that("works for datasets", {

  ds <- torch::dataset(
    initialize = function() {},
    .getitem = function(i) {
      torch::torch_randn(10, 10)
    },
    .length = function() {100}
  )

  d <- as_dataloader(ds())
  x <- coro::collect(d)

  expect_length(x, 4)
  expect_tensor_shape(x[[1]], c(32, 10, 10))

  d <- as_dataloader(ds(), batch_size = 10)
  expect_equal(length(d), 10)

  x <- coro::collect(d)

  expect_length(x, 10)
  expect_tensor_shape(x[[1]], c(10, 10, 10))

})

test_that("works for lists of tensors", {

  ds <- list(torch::torch_randn(100, 10), torch::torch_randn(100, 5))
  d <- as_dataloader(ds)

  x <- coro::collect(d)
  expect_length(x, 4)
  expect_tensor_shape(x[[1]][[1]], c(32, 10))
  expect_tensor_shape(x[[1]][[2]], c(32, 5))

  ds <- list(torch::torch_randn(100, 10), array(1, dim = c(100, 5)))
  d <- as_dataloader(ds)

  x <- coro::collect(d)
  expect_length(x, 4)
  expect_tensor_shape(x[[1]][[1]], c(32, 10))
  expect_tensor_shape(x[[1]][[2]], c(32, 5))

})

test_that("works for nuemrics", {

  x <- runif(100)
  l <- as_dataloader(x)
  expect_equal(length(l), 4)

  x <- matrix(runif(1000), ncol = 10)
  l <- as_dataloader(x)
  expect_equal(length(l), 4)

  x <- array(runif(1000), dim = c(100, 10, 10))
  l <- as_dataloader(x)
  expect_equal(length(l), 4)

  x <- torch::torch_randn(100, 10)
  l <- as_dataloader(x)
  expect_equal(length(l), 4)

})
