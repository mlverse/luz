# Set hyper-parameter of a module

This function is used to define hyper-parameters before calling `fit`
for `luz_modules`.

## Usage

``` r
set_hparams(module, ...)
```

## Arguments

- module:

  An `nn_module` that has been
  [`setup()`](https://mlverse.github.io/luz/reference/setup.md).

- ...:

  The parameters set here will be used to initialize the `nn_module`, ie
  they are passed unchanged to the `initialize` method of the base
  `nn_module`.

## Value

The same luz module

## See also

Other set_hparam:
[`set_opt_hparams()`](https://mlverse.github.io/luz/reference/set_opt_hparams.md)
