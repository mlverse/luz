library(torch)

test_that("Device dataloader", {

  x <- tensor_dataset(x = torch_randn(100, 10), y = torch_randn(100, 15))
  dl <- dataloader(x, batch_size = 10, shuffle = FALSE)

  d <- as_device_dataloader(dl, "cpu")
  coro::loop(for (i in dl) {
    expect_equal(i$x$shape, c(10, 10))
    expect_equal(i$y$shape, c(10, 15))
  })

  expect_equal(as.array(i$x), as.array(x[91:100]$x))
  expect_equal(as.array(i$y), as.array(x[91:100]$y))

})

test_that("switch parameters", {

  if (!cuda_is_available())
    skip("CUDA not available.")

  model <- nn_linear(10, 10)
  old <- get_parameter_ids(model)

  opt <- optim_sgd(model$parameters, lr = 0.1)
  model$to(device = "cuda")

  new <- get_parameter_ids(model)
  mapping <- setNames(new, names(old))
  switch_parameters(opt, .mapping = mapping)

  expect_equal(
    opt$param_groups[[1]]$params[[1]]$storage()$data_ptr(),
    model$parameters[[1]]$storage()$data_ptr()
  )
})
