test_that("label smoothing works", {

  output <- torch::torch_randn(32, 5, 10, device = get_device())
  target <- torch::torch_randint(1, 10, size = list(32, 5), device = get_device())

  lmce <- luz_label_smoothing_cross_entropy(output$flatten(start_dim = 1, end_dim = 2), target$flatten())
  lmce_t <- luz_label_smoothing_cross_entropy(output$transpose(dim0 = 3, dim1 = 2), target)

  expect_equal(as.numeric(lmce$to(device = "cpu")), as.numeric(lmce_t$to(device = "cpu")))
})
