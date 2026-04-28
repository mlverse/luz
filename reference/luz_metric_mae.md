# Mean absolute error

Computes the mean absolute error.

## Usage

``` r
luz_metric_mae()
```

## Value

Returns new luz metric.

## See also

Other luz_metrics:
[`luz_metric()`](https://mlverse.github.io/luz/reference/luz_metric.md),
[`luz_metric_accuracy()`](https://mlverse.github.io/luz/reference/luz_metric_accuracy.md),
[`luz_metric_binary_accuracy()`](https://mlverse.github.io/luz/reference/luz_metric_binary_accuracy.md),
[`luz_metric_binary_accuracy_with_logits()`](https://mlverse.github.io/luz/reference/luz_metric_binary_accuracy_with_logits.md),
[`luz_metric_binary_auroc()`](https://mlverse.github.io/luz/reference/luz_metric_binary_auroc.md),
[`luz_metric_mse()`](https://mlverse.github.io/luz/reference/luz_metric_mse.md),
[`luz_metric_multiclass_auroc()`](https://mlverse.github.io/luz/reference/luz_metric_multiclass_auroc.md),
[`luz_metric_rmse()`](https://mlverse.github.io/luz/reference/luz_metric_rmse.md)

## Examples

``` r
if (torch::torch_is_installed()) {
library(torch)
metric <- luz_metric_mae()
metric <- metric$new()
metric$update(torch_randn(100), torch_randn(100))
metric$compute()
}
#> [1] 1.016295
```
