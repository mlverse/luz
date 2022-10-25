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

expect_recursive_equal <- function(x, y) {
  if (is.list(x))
    map2(x, y, expect_recursive_equal)
  else if (inherits(x, "torch_tensor"))
    expect_equal_to_tensor(x, y)
  else
    expect_equal(x, y)
}
