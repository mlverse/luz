# Computes the multi-class AUROC

The same definition as
[Keras](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC)
is used by default. This is equivalent to the `'micro'` method in SciKit
Learn too. See
[docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).

## Usage

``` r
luz_metric_multiclass_auroc(
  num_thresholds = 200,
  thresholds = NULL,
  from_logits = FALSE,
  average = c("micro", "macro", "weighted", "none")
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

  If `TRUE` then we call
  [`torch::nnf_softmax()`](https://torch.mlverse.org/docs/reference/nnf_softmax.html)
  in the predictions before computing the metric.

- average:

  The averaging method:

  - `'micro'`: Stack all classes and computes the AUROC as if it was a
    binary classification problem.

  - `'macro'`: Finds the AUCROC for each class and computes their mean.

  - `'weighted'`: Finds the AUROC for each class and computes their
    weighted mean pondering by the number of instances for each class.

  - `'none'`: Returns the AUROC for each class in a list.

## Details

**Note** that class imbalance can affect this metric unlike the AUC for
binary classification.

Currently the AUC is approximated using the 'interpolation' method
described in
[Keras](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC).

## See also

Other luz_metrics:
[`luz_metric()`](https://mlverse.github.io/luz/reference/luz_metric.md),
[`luz_metric_accuracy()`](https://mlverse.github.io/luz/reference/luz_metric_accuracy.md),
[`luz_metric_binary_accuracy()`](https://mlverse.github.io/luz/reference/luz_metric_binary_accuracy.md),
[`luz_metric_binary_accuracy_with_logits()`](https://mlverse.github.io/luz/reference/luz_metric_binary_accuracy_with_logits.md),
[`luz_metric_binary_auroc()`](https://mlverse.github.io/luz/reference/luz_metric_binary_auroc.md),
[`luz_metric_mae()`](https://mlverse.github.io/luz/reference/luz_metric_mae.md),
[`luz_metric_mse()`](https://mlverse.github.io/luz/reference/luz_metric_mse.md),
[`luz_metric_rmse()`](https://mlverse.github.io/luz/reference/luz_metric_rmse.md)

## Examples

``` r
if (torch::torch_is_installed()) {
library(torch)
actual <- c(1, 1, 1, 0, 0, 0) + 1L
predicted <- c(0.9, 0.8, 0.4, 0.5, 0.3, 0.2)
predicted <- cbind(1-predicted, predicted)

y_true <- torch_tensor(as.integer(actual))
y_pred <- torch_tensor(predicted)

m <- luz_metric_multiclass_auroc(thresholds = as.numeric(predicted),
                                 average = "micro")
m <- m$new()

m$update(y_pred[1:2,], y_true[1:2])
m$update(y_pred[3:4,], y_true[3:4])
m$update(y_pred[5:6,], y_true[5:6])
m$compute()
}
#> [1] 0.9027778
```
