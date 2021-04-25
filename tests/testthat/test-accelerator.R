library(torch)

test_that("Device dataloader", {

  x <- tensor_dataset(x = torch_ones(100, 10), y = torch_ones(100, 15))
  dl <- dataloader(x, batch_size = 10)

  d <- as_device_dataloader(dl, "cpu")
  coro::loop(for (i in dl) {
    expect_equal(i$x$shape, c(10, 10))
    expect_equal(i$y$shape, c(10, 15))
  })
})
