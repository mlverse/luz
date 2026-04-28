# Computes the area under the ROC

To avoid storing all predictions and targets for an epoch we compute
confusion matrices across a range of pre-established thresholds.

## Usage

``` r
luz_metric_binary_auroc(
  num_thresholds = 200,
  thresholds = NULL,
  from_logits = FALSE
)
```

## Arguments

- num_thresholds:

  Number of thresholds used to compute confusion matrices. In that case,
  thresholds are created by getting `num_thresholds` values linearly
  spaced in the unit interval.

- thresholds:

  (optional) If threshold are passed, then those are used to compute the
  confusion matrices and `num_thresholds` is ignored.

- from_logits:

  Boolean indicating if predictions are logits, in that case we use
  sigmoid to put them in the unit interval.

## See also

Other luz_metrics:
[`luz_metric()`](https://mlverse.github.io/luz/reference/luz_metric.md),
[`luz_metric_accuracy()`](https://mlverse.github.io/luz/reference/luz_metric_accuracy.md),
[`luz_metric_binary_accuracy()`](https://mlverse.github.io/luz/reference/luz_metric_binary_accuracy.md),
[`luz_metric_binary_accuracy_with_logits()`](https://mlverse.github.io/luz/reference/luz_metric_binary_accuracy_with_logits.md),
[`luz_metric_mae()`](https://mlverse.github.io/luz/reference/luz_metric_mae.md),
[`luz_metric_mse()`](https://mlverse.github.io/luz/reference/luz_metric_mse.md),
[`luz_metric_multiclass_auroc()`](https://mlverse.github.io/luz/reference/luz_metric_multiclass_auroc.md),
[`luz_metric_rmse()`](https://mlverse.github.io/luz/reference/luz_metric_rmse.md)

## Examples

``` r
if (torch::torch_is_installed()){
library(torch)
actual <- c(1, 1, 1, 0, 0, 0)
predicted <- c(0.9, 0.8, 0.4, 0.5, 0.3, 0.2)

y_true <- torch_tensor(actual)
y_pred <- torch_tensor(predicted)

m <- luz_metric_binary_auroc(thresholds = predicted)
m <- m$new()

m$update(y_pred[1:2], y_true[1:2])
m$update(y_pred[3:4], y_true[3:4])
m$update(y_pred[5:6], y_true[5:6])

m$compute()
}
#> [1] 0.8888889
```
