# Loss to be used with `callbacks_mixup()`.

In the training phase, computes individual losses with regard to two
targets, weights them item-wise, and averages the linear combinations to
yield the mean batch loss. For validation and testing, defers to the
passed-in loss.

## Usage

``` r
nn_mixup_loss(loss)
```

## Arguments

- loss:

  the underlying loss `nn_module` to call. It must support the
  `reduction` field. During training the attribute will be changed to
  `'none'` so we get the loss for individual observations. See for for
  example documentation for the `reduction` argument in
  [`torch::nn_cross_entropy_loss()`](https://torch.mlverse.org/docs/reference/nn_cross_entropy_loss.html).

## Details

It should be used together with
[`luz_callback_mixup()`](https://mlverse.github.io/luz/reference/luz_callback_mixup.md).

## See also

[`luz_callback_mixup()`](https://mlverse.github.io/luz/reference/luz_callback_mixup.md)
