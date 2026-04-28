# Set optimizer hyper-parameters

This function is used to define hyper-parameters for the optimizer
initialization method.

## Usage

``` r
set_opt_hparams(module, ...)
```

## Arguments

- module:

  An `nn_module` that has been
  [`setup()`](https://mlverse.github.io/luz/reference/setup.md).

- ...:

  The parameters passed here will be used to initialize the optimizers.
  For example, if your optimizer is `optim_adam` and you pass `lr=0.1`,
  then the `optim_adam` function is called with
  `optim_adam(parameters, lr=0.1)` when fitting the model.

## Value

The same luz module

## See also

Other set_hparam:
[`set_hparams()`](https://mlverse.github.io/luz/reference/set_hparams.md)
