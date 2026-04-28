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
#> Columns 1 to 6-2.7339e-01 -4.3452e-01  1.1065e+00 -2.1347e-01 -6.4266e-01  1.4738e+00
#> -9.0977e-01 -8.0733e-01  6.5917e-01 -4.8101e-01 -1.3594e+00  1.1691e+00
#>  2.3583e+00 -8.5315e-01 -7.7135e-01  4.6731e-01 -2.0571e+00 -5.4393e-01
#>  3.9808e-01 -6.2180e-01 -4.0482e-02  9.9698e-01 -5.9495e-01 -5.3886e-01
#>  6.7323e-01 -1.3305e+00  9.6319e-01  6.3976e-01 -4.8898e-01 -3.8634e-01
#>  2.7454e-01 -3.7022e-01  9.7635e-01 -3.6721e-01  1.2811e+00 -1.4694e-01
#> -7.1102e-01  1.7116e-01 -1.8145e-01 -4.8933e-01  5.2694e-01  1.9816e-01
#>  1.6816e-01 -5.1300e-01 -2.1990e-01  1.1283e+00  4.2140e-01 -9.8432e-01
#>  1.4914e+00  5.6667e-01  4.1093e-01  2.9734e-01 -1.9747e+00  9.6860e-01
#>  6.4067e-01  7.6104e-01  1.4900e+00  6.4677e-01 -1.5932e+00 -4.3372e-04
#> 
#> Columns 7 to 12 1.3483e+00  2.3311e-03 -1.5799e+00  1.6648e+00  8.3211e-01 -9.6291e-01
#> -9.9579e-02  9.6863e-01  1.5427e-01  3.5612e-01  4.7892e-01 -3.2170e-01
#>  1.6774e+00  3.3094e-02  4.6043e-01 -3.6044e-01  1.4677e-01 -5.9293e-01
#> -1.1278e+00 -3.1448e-01 -3.2600e-01 -2.3456e-02  2.1046e-01 -1.0177e+00
#>  2.5115e+00  5.0055e-02  1.2502e+00  4.9242e-01  7.3080e-01 -5.1545e-01
#>  9.9906e-02 -7.0031e-01  1.5770e+00 -4.4589e-01  1.3089e+00  8.9109e-01
#>  6.9131e-01 -1.7504e-01 -3.9118e-01 -1.4200e+00  7.5280e-01 -2.6933e-02
#>  5.0142e-01 -8.9373e-01 -9.7049e-01 -1.0340e+00  1.9294e+00 -6.7376e-01
#>  1.4730e-02 -4.0475e-01  4.2358e-01  5.5356e-01  8.0452e-01  1.1909e+00
#> -5.1276e-01  1.3366e+00  6.6148e-01 -1.3622e+00 -7.4601e-01  2.0181e-02
#> 
#> Columns 13 to 18-1.2373e+00 -1.2669e+00 -1.4861e+00 -1.6657e+00  2.6566e-01 -2.7482e+00
#>  9.2684e-01  7.3976e-01  7.6841e-01  1.0315e+00 -3.6030e-01  2.6102e-01
#>  2.5143e-01  1.6068e-01 -6.5582e-01  3.9535e-01 -7.5355e-01 -1.8086e+00
#> -6.8676e-01  3.5877e-01 -3.3061e-01  5.3759e-01 -1.0502e+00 -2.2570e-01
#>  6.3524e-01 -7.3081e-01 -1.6477e+00 -4.2382e-01  6.9011e-01 -5.3982e-01
#>  8.5584e-01 -1.1197e+00 -6.3201e-01 -5.2269e-01 -3.5144e-01  7.3074e-01
#> -9.4508e-02  9.1412e-01 -1.9171e+00 -1.1710e+00  1.1097e+00 -3.5775e-01
#>  1.8693e+00 -6.0363e-01 -7.8081e-02  3.4981e-01 -9.9513e-01  1.0351e+00
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,768} ]
#> 
#> $y
#> $y$ys
#> $y$ys$y1
#> torch_tensor
#> -0.3776
#>  1.2264
#>  1.5508
#> -0.8340
#>  0.2476
#>  1.6801
#> -0.5425
#> -0.3458
#> -0.9292
#>  1.2079
#> [ CPUFloatType{10} ]
#> 
#> $y$ys$y2
#> torch_tensor
#> -0.3776
#>  0.2476
#>  1.2079
#> -0.5425
#>  1.6801
#> -0.9292
#> -0.3458
#> -0.8340
#>  1.5508
#>  1.2264
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
