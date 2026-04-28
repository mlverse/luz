# Mixup logic

Logic underlying
[`luz_callback_mixup()`](https://mlverse.github.io/luz/reference/luz_callback_mixup.md).

## Usage

``` r
nnf_mixup(x, y, weight)
```

## Arguments

- x:

  an input batch

- y:

  a target batch

- weight:

  weighting coefficient to be used by
  [`torch_lerp()`](https://torch.mlverse.org/docs/reference/torch_lerp.html)

## Value

A `list` of:

- `x`, the new, mixed-up input batch

- `y`, a `list` of:

  - `ys`, a `list` of:

    - `y1`, the original target `y1`

    - `y2`, the mixed-in target `y2`

  - `weight`, the mixing weights

## Details

Based on the passed-in input and target batches, as well as applicable
mixing weights, we return new tensors intended to replace the current
batch. The new input batch is a weighted linear combination of input
batch items, while the new target batch bundles the original targets, as
well as the mixing weights, in a nested list.

## See also

[`luz_callback_mixup()`](https://mlverse.github.io/luz/reference/luz_callback_mixup.md)

## Examples

``` r
if (torch::torch_is_installed()) {
batch_x <- torch::torch_randn(c(10, 768))
batch_y <- torch::torch_randn(10)
weight <- torch::torch_tensor(rep(0.9, 10))$view(c(10, 1))
nnf_mixup(batch_x, batch_y, weight)
}
#> $x
#> torch_tensor
#> Columns 1 to 6-4.8621e-01 -1.5327e+00 -3.8725e-01  9.7633e-01  4.0168e-01 -6.4193e-01
#> -1.0477e+00 -1.8631e-01 -1.1509e+00 -7.1780e-02  9.2415e-01 -1.2145e+00
#> -1.6780e-01 -1.1256e+00  9.0943e-02 -1.9971e-01 -2.5206e-01  2.3414e-01
#>  4.7140e-01 -8.5727e-01 -1.7073e+00  1.9943e+00  4.1375e-01 -1.5723e+00
#> -1.0172e-01 -3.6266e-01 -1.1818e+00  1.5315e+00  1.7207e+00  3.5827e-01
#>  3.5826e-01 -6.1648e-01  1.2056e-01 -6.8369e-01  8.4748e-01  1.5345e+00
#> -7.7667e-02 -5.2315e-02 -2.9107e-01 -3.6411e-01  8.8496e-01 -1.5857e+00
#>  3.0101e-01  5.7758e-01  9.3187e-01  8.3876e-01 -3.1853e-01  1.2825e+00
#> -7.3471e-01  9.5928e-01 -9.7608e-01  4.7876e-01  1.6025e+00  7.5702e-01
#> -2.1694e+00  9.0505e-01  1.8209e-01  1.0872e+00 -9.1756e-01  5.7291e-01
#> 
#> Columns 7 to 12-1.4400e+00 -5.2230e-01 -6.5743e-01 -2.9252e-01 -5.9598e-02 -1.2691e+00
#>  5.4092e-01  3.0664e-01  9.3532e-02  1.4724e+00  5.4040e-01 -4.5337e-01
#> -2.8815e-01 -1.5170e+00  3.5313e-01 -5.4612e-01 -5.8500e-01 -1.1607e-01
#> -1.1626e+00  8.1915e-01 -2.5632e-01 -9.5961e-01  5.3190e-01 -4.9549e-01
#> -1.0095e+00 -7.8346e-01  1.8561e+00 -1.1493e+00 -4.9622e-01 -8.6298e-01
#>  5.2372e-01  7.0657e-01  1.3405e+00 -5.4801e-01 -8.7372e-02  4.7652e-02
#> -8.9105e-01 -1.7355e-01 -8.0010e-01 -6.7034e-01 -1.1654e+00  1.0177e-01
#> -6.2471e-03 -1.8025e+00  1.8888e+00 -1.1897e+00  1.0813e+00  1.7793e+00
#>  4.3846e-01 -6.6269e-01 -2.6436e-01  2.7748e-01 -1.7485e+00 -5.8658e-02
#>  4.8779e-01  2.2038e-01  4.8830e-01  9.8358e-01  3.3080e-01 -7.2176e-01
#> 
#> Columns 13 to 18-2.3198e-01  4.5457e-01  4.7395e-02  4.4841e-01 -1.9146e+00  1.2983e+00
#> -1.8780e+00 -1.1890e+00 -4.4058e-01  4.1797e-01 -2.8347e-01 -1.9255e-01
#> -1.6993e+00  7.1852e-01 -5.9666e-01 -8.4062e-01  2.5406e-01 -1.1154e+00
#> -1.0818e+00 -6.0030e-01  8.9602e-01 -1.3250e+00 -1.2331e+00 -9.8670e-01
#> -1.0497e+00  9.4995e-01 -3.0696e-01 -4.6534e-01 -7.5936e-02  5.2797e-01
#> -5.2245e-01 -3.0039e-01 -2.5332e-01 -1.0996e-01 -4.6102e-01  2.8179e-01
#> -1.8453e-01  6.5629e-01 -6.6929e-01 -2.5970e-01  4.6935e-01  8.6960e-02
#> -1.4410e+00  9.4411e-01 -3.0762e-01 -8.1687e-01  1.5169e+00 -2.5810e-02
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,768} ]
#> 
#> $y
#> $y$ys
#> $y$ys$y1
#> torch_tensor
#>  1.7929
#> -0.7193
#>  0.4292
#>  0.0940
#>  1.3912
#>  0.5608
#> -0.0052
#>  0.3561
#>  0.4418
#>  1.3434
#> [ CPUFloatType{10} ]
#> 
#> $y$ys$y2
#> torch_tensor
#> -0.7193
#>  0.4418
#>  0.5608
#>  1.3912
#>  1.7929
#> -0.0052
#>  0.4292
#>  0.0940
#>  0.3561
#>  1.3434
#> [ CPUFloatType{10} ]
#> 
#> 
#> $y$weight
#> torch_tensor
#>  0.9000
#>  0.9000
#>  0.9000
#>  0.9000
#>  0.9000
#>  0.9000
#>  0.9000
#>  0.9000
#>  0.9000
#>  0.9000
#> [ CPUFloatType{10,1} ]
#> 
#> 
```
