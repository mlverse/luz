# Accuracy

Computes accuracy for multi-class classification problems.

## Usage

``` r
luz_metric_accuracy()
```

## Value

Returns new luz metric.

## Details

This metric expects to take logits or probabilities at every update. It
will then take the columnwise argmax and compare to the target.

## See also

Other luz_metrics:
[`luz_metric()`](https://mlverse.github.io/luz/reference/luz_metric.md),
[`luz_metric_binary_accuracy()`](https://mlverse.github.io/luz/reference/luz_metric_binary_accuracy.md),
[`luz_metric_binary_accuracy_with_logits()`](https://mlverse.github.io/luz/reference/luz_metric_binary_accuracy_with_logits.md),
[`luz_metric_binary_auroc()`](https://mlverse.github.io/luz/reference/luz_metric_binary_auroc.md),
[`luz_metric_mae()`](https://mlverse.github.io/luz/reference/luz_metric_mae.md),
[`luz_metric_mse()`](https://mlverse.github.io/luz/reference/luz_metric_mse.md),
[`luz_metric_multiclass_auroc()`](https://mlverse.github.io/luz/reference/luz_metric_multiclass_auroc.md),
[`luz_metric_rmse()`](https://mlverse.github.io/luz/reference/luz_metric_rmse.md)

## Examples

``` r
if (torch::torch_is_installed()) {
library(torch)
metric <- luz_metric_accuracy()
metric <- metric$new()
metric$update(torch_randn(100, 10), torch::torch_randint(1, 10, size = 100))
metric$compute()
}
#> [1] 0.11
```
