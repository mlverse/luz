test_that("binary auc works", {

  actual <- c(1, 1, 1, 0, 0, 0)
  predicted <- c(0.9, 0.8, 0.4, 0.5, 0.3, 0.2)

  y_true <- torch_tensor(actual, device = get_device())
  y_pred <- torch_tensor(predicted, device = get_device())

  m <- luz_metric_binary_auroc(thresholds = predicted)
  m <- m$new()$to(device = get_device())

  m$update(y_pred[1:2], y_true[1:2])
  m$update(y_pred[3:4], y_true[3:4])
  m$update(y_pred[5:6], y_true[5:6])

  expect_equal(
    m$compute(),
    Metrics::auc(actual, predicted)
  )

})

test_that("multiclass auc works with method = micro", {

  actual <- c(1, 1, 1, 0, 0, 0) + 1L
  predicted <- c(0.9, 0.8, 0.4, 0.5, 0.3, 0.2)
  predicted <- cbind(1-predicted, predicted)

  y_true <- torch_tensor(as.integer(actual), device = get_device())
  y_pred <- torch_tensor(predicted, device = get_device())

  m <- luz_metric_multiclass_auroc(thresholds = as.numeric(predicted),
                                   average = "micro")
  m <- m$new()$to(device = get_device())

  m$update(y_pred[1:2,], y_true[1:2])
  m$update(y_pred[3:4,], y_true[3:4])
  m$update(y_pred[5:6,], y_true[5:6])

  expect_equal(
    m$compute(),
    Metrics::auc(
      as.numeric(model.matrix(~0 + as.factor(actual))),
      as.numeric(predicted)
    )
  )

})

test_that("multiclass auc works with method = macro", {

  actual <- c(1, 1, 1, 0, 0, 0) + 1L
  predicted <- c(0.9, 0.8, 0.4, 0.5, 0.3, 0.2)
  predicted <- cbind(1-predicted, predicted)

  y_true <- torch_tensor(as.integer(actual), device = get_device())
  y_pred <- torch_tensor(predicted, device = get_device())

  m <- luz_metric_multiclass_auroc(num_thresholds = 1e5, average = "macro")
  m <- m$new()$to(device = get_device())

  m$update(y_pred[1:2,], y_true[1:2])
  m$update(y_pred[3:4,], y_true[3:4])
  m$update(y_pred[5:6,], y_true[5:6])

  expect_equal(
    m$compute(),
    mean(c(
      Metrics::auc(as.numeric(actual == 1), predicted[,1]),
      Metrics::auc(as.numeric(actual == 2), predicted[,2])
    ))
  )

})

test_that("multiclass auc works with method = weighted", {

  actual <- c(1, 1, 1, 0, 0, 0, 1, 1) + 1L
  predicted <- c(0.9, 0.8, 0.4, 0.5, 0.3, 0.2, 0.4, 0.3)
  predicted <- cbind(predicted, predicted)

  y_true <- torch_tensor(as.integer(actual), device = get_device())
  y_pred <- torch_tensor(predicted, device = get_device())

  m <- luz_metric_multiclass_auroc(num_thresholds = 1e5, average = "weighted")
  m <- m$new()$to(device = get_device())

  m$update(y_pred[1:2,], y_true[1:2])
  m$update(y_pred[3:4,], y_true[3:4])
  m$update(y_pred[5:6,], y_true[5:6])
  m$update(y_pred[7:8,], y_true[7:8])

  expect_equal(
    m$compute(),
    mean(actual == 1) * Metrics::auc(as.numeric(actual == 1), predicted[,1]) +
      mean(actual == 2) * Metrics::auc(as.numeric(actual == 2), predicted[,2])
  )

})

test_that("multiclass auc works with `none`", {

  actual <- c(1, 1, 1, 0, 0, 0) + 1L
  predicted <- c(0.9, 0.8, 0.4, 0.5, 0.3, 0.2)
  predicted <- cbind(1-predicted, predicted)

  y_true <- torch_tensor(as.integer(actual), device = get_device())
  y_pred <- torch_tensor(predicted, device = get_device())

  m <- luz_metric_multiclass_auroc(num_thresholds = 1e5, average = "none")
  m <- m$new()$to(device = get_device())

  m$update(y_pred[1:2,], y_true[1:2])
  m$update(y_pred[3:4,], y_true[3:4])
  m$update(y_pred[5:6,], y_true[5:6])

  expect_equal(
    m$compute(),
    list(
      Metrics::auc(as.numeric(actual == 1), predicted[,1]),
      Metrics::auc(as.numeric(actual == 2), predicted[,2])
    )
  )

})



