test_that("binary auc works", {

  actual <- c(1, 1, 1, 0, 0, 0)
  predicted <- c(0.9, 0.8, 0.4, 0.5, 0.3, 0.2)

  y_true <- torch_tensor(actual)
  y_pred <- torch_tensor(predicted)

  m <- luz_metric_binary_auroc(thresholds = predicted)
  m <- m$new()

  m$update(y_pred[1:2], y_true[1:2])
  m$update(y_pred[3:4], y_true[3:4])
  m$update(y_pred[5:6], y_true[5:6])

  expect_equal(
    m$compute(),
    Metrics::auc(actual, predicted)
  )

})

test_that("multiclass auc works", {

  actual <- c(1, 1, 1, 0, 0, 0) + 1L
  predicted <- c(0.9, 0.8, 0.4, 0.5, 0.3, 0.2)
  predicted <- cbind(1-predicted, predicted)

  y_true <- torch_tensor(as.integer(actual))
  y_pred <- torch_tensor(predicted)

  m <- luz_metric_multiclass_auc(num_thresholds = 1e5)
  m <- m$new()

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
