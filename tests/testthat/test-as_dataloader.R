test_that("as_dataloader fails", {
  t <- structure(1, class = c("test", "hi"))
  expect_error(
    as_dataloader(t),
    class = "value_error"
  )
})
