# Set's up a `nn_module` to use with luz

The setup function is used to set important attributes and method for
`nn_modules` to be used with luz.

## Usage

``` r
setup(module, loss = NULL, optimizer = NULL, metrics = NULL, backward = NULL)
```

## Arguments

- module:

  (`nn_module`) The `nn_module` that you want set up.

- loss:

  (`function`, optional) An optional function with the signature
  `function(input, target)`. It's only requires if your `nn_module`
  doesn't implement a method called `loss`.

- optimizer:

  (`torch_optimizer`, optional) A function with the signature
  `function(parameters, ...)` that is used to initialize an optimizer
  given the model parameters.

- metrics:

  (`list`, optional) A list of metrics to be tracked during the training
  procedure. Sometimes, you want some metrics to be evaluated only
  during training or validation, in this case you can pass a
  [`luz_metric_set()`](https://mlverse.github.io/luz/reference/luz_metric_set.md)
  object to specify metrics used in each stage.

- backward:

  (`function`) A functions that takes the loss scalar values as it's
  parameter. It must call `$backward()` or
  [`torch::autograd_backward()`](https://torch.mlverse.org/docs/reference/autograd_backward.html).
  In general you don't need to set this parameter unless you need to
  customize how luz calls the `backward()`, for example, if you need to
  add additional arguments to the backward call. Note that this becomes
  a method of the `nn_module` thus can be used by your custom
  [`step()`](https://rdrr.io/r/stats/step.html) if you override it.

## Value

A luz module that can be trained with
[`fit()`](https://generics.r-lib.org/reference/fit.html).

## Details

It makes sure the module have all the necessary ingredients in order to
be fitted.

## Note

It also adds a `device` active field that can be used to query the
current module `device` within methods, with eg `self$device`. This is
useful when [`ctx()`](https://mlverse.github.io/luz/reference/ctx.md) is
not available, eg, when calling methods from outside the `luz` wrappers.
Users can override the default by implementing a `device` active method
in the input `module`.

## See also

Other training:
[`evaluate()`](https://mlverse.github.io/luz/reference/evaluate.md),
[`fit.luz_module_generator()`](https://mlverse.github.io/luz/reference/fit.luz_module_generator.md),
[`predict.luz_module_fitted()`](https://mlverse.github.io/luz/reference/predict.luz_module_fitted.md)
