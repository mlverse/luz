test_that("nn_mixup_loss works when used with cross entropy", {

  # 1: manual check for constant mixing weight of 0.9
  target1 <- torch::torch_ones(7, dtype = torch::torch_long())
  target2 <- torch::torch_empty_like(target1)$fill_(4)
  weight <- torch::torch_empty_like(target1, dtype = torch::torch_float())$fill_(0.9)
  target <- list(list(target1, target2), weight)
  input <- torch::torch_randn(c(7, 4))

  mixup_loss <- nn_mixup_loss(torch::nn_cross_entropy_loss(ignore_index = 222))
  t1_loss <- torch::nnf_cross_entropy(input, target1, reduction = "none")
  t2_loss <- torch::nnf_cross_entropy(input, target2, reduction = "none")
  expect_equal(as.numeric(mixup_loss(input, target)), as.numeric(torch::torch_mean(t1_loss + 0.9 * (t2_loss - t1_loss))), tolerance = 1e-4)

  # 2: mixing weight of 1 yields same loss as using target2 only
  weight <- torch::torch_empty_like(target1, dtype = torch::torch_float())$fill_(1)
  target <- list(list(target1, target2), weight)
  mixup_loss <- nn_mixup_loss(torch::nn_cross_entropy_loss())
  t2_loss <- torch::nnf_cross_entropy(input, target2)
  expect_equal(as.numeric(mixup_loss(input, target)), as.numeric(t2_loss), tolerance = 1e-6)

})
