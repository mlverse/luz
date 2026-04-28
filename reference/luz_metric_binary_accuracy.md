# Binary accuracy

Computes the accuracy for binary classification problems where the model
returns probabilities. Commonly used when the loss is
[`torch::nn_bce_loss()`](https://torch.mlverse.org/docs/reference/nn_bce_loss.html).

## Usage

``` r
luz_metric_binary_accuracy(threshold = 0.5)
```

## Arguments

- threshold:

  value used to classifiy observations between 0 and 1.

## Value

Returns new luz metric.

## See also

Other luz_metrics:
[`luz_metric()`](https://mlverse.github.io/luz/reference/luz_metric.md),
[`luz_metric_accuracy()`](https://mlverse.github.io/luz/reference/luz_metric_accuracy.md),
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
metric <- luz_metric_binary_accuracy(threshold = 0.5)
metric <- metric$new()
metric$update(torch_rand(100), torch::torch_randint(0, 1, size = 100))
metric$compute()
}
#> [1] 0.44
```
