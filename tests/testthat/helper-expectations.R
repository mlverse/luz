options("torch.old_seed_behavior" = TRUE)

expect_equal_to_tensor <- function(object, expected, ...) {
  expect_tensor(object)
  expect_tensor(expected)

  expect_true(torch::torch_allclose(object, expected, ...))
}

expect_not_equal_to_tensor <- function(object, expected, ...) {
  expect_tensor(object)
  expect_tensor(expected)

  expect_true(!torch::torch_allclose(object, expected, ...))
}

expect_tensor <- function(object) {
  expect_true(inherits(object, "torch_tensor"))
}

expect_tensor_shape <- function(object, shape) {
  expect_tensor(object)
  expect_equal(object$shape, shape)
}
