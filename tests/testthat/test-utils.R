test_that("bind_context works for more complicated classes", {

  x <- R6::R6Class("hello",public = list(get_x = function() ctx$x))
  y <- R6::R6Class(inherit = x)

  instance <- y$new()
  ctx <- context$new()
  bind_context(instance, ctx)

  expect_equal(instance$get_x(), ctx$x)
})
