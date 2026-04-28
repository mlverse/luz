# Binary accuracy with logits

Computes accuracy for binary classification problems where the model
return logits. Commonly used together with
[`torch::nn_bce_with_logits_loss()`](https://torch.mlverse.org/docs/reference/nn_bce_with_logits_loss.html).

## Usage

``` r
luz_metric_binary_accuracy_with_logits(threshold = 0.5)
```

## Arguments

- threshold:

  value used to classifiy observations between 0 and 1.

## Value

Returns new luz metric.

## Details

Probabilities are generated using
[`torch::nnf_sigmoid()`](https://torch.mlverse.org/docs/reference/nnf_sigmoid.html)
and `threshold` is used to classify between 0 or 1.

## See also

Other luz_metrics:
[`luz_metric()`](https://mlverse.github.io/luz/reference/luz_metric.md),
[`luz_metric_accuracy()`](https://mlverse.github.io/luz/reference/luz_metric_accuracy.md),
[`luz_metric_binary_accuracy()`](https://mlverse.github.io/luz/reference/luz_metric_binary_accuracy.md),
[`luz_metric_binary_auroc()`](https://mlverse.github.io/luz/reference/luz_metric_binary_auroc.md),
[`luz_metric_mae()`](https://mlverse.github.io/luz/reference/luz_metric_mae.md),
[`luz_metric_mse()`](https://mlverse.github.io/luz/reference/luz_metric_mse.md),
[`luz_metric_multiclass_auroc()`](https://mlverse.github.io/luz/reference/luz_metric_multiclass_auroc.md),
[`luz_metric_rmse()`](https://mlverse.github.io/luz/reference/luz_metric_rmse.md)

## Examples

``` r
if (torch::torch_is_installed()) {
library(torch)
metric <- luz_metric_binary_accuracy_with_logits(threshold = 0.5)
metric <- metric$new()
metric$update(torch_randn(100), torch::torch_randint(0, 1, size = 100))
metric$compute()
}
#> [1] 0.54
```
