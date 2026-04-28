# Create predictions for a fitted model

Create predictions for a fitted model

## Usage

``` r
# S3 method for class 'luz_module_fitted'
predict(
  object,
  newdata,
  ...,
  callbacks = list(),
  accelerator = NULL,
  verbose = NULL,
  dataloader_options = NULL
)
```

## Arguments

- object:

  (fitted model) the fitted model object returned from
  [`fit.luz_module_generator()`](https://mlverse.github.io/luz/reference/fit.luz_module_generator.md)

- newdata:

  (dataloader, dataset, list or array) returning a list with at least 1
  element. The other elements aren't used.

- ...:

  Currently unused.

- callbacks:

  (list, optional) A list of callbacks defined with
  [`luz_callback()`](https://mlverse.github.io/luz/reference/luz_callback.md)
  that will be called during the training procedure. The callbacks
  [`luz_callback_metrics()`](https://mlverse.github.io/luz/reference/luz_callback_metrics.md),
  [`luz_callback_progress()`](https://mlverse.github.io/luz/reference/luz_callback_progress.md)
  and
  [`luz_callback_train_valid()`](https://mlverse.github.io/luz/reference/luz_callback_train_valid.md)
  are always added by default.

- accelerator:

  (accelerator, optional) An optional
  [`accelerator()`](https://mlverse.github.io/luz/reference/accelerator.md)
  object used to configure device placement of the components like
  [torch::nn_module](https://torch.mlverse.org/docs/reference/nn_module.html)s,
  optimizers and batches of data.

- verbose:

  (logical, optional) An optional boolean value indicating if the
  fitting procedure should emit output to the console during training.
  By default, it will produce output if
  [`interactive()`](https://rdrr.io/r/base/interactive.html) is `TRUE`,
  otherwise it won't print to the console.

- dataloader_options:

  Options used when creating a dataloader. See
  [`torch::dataloader()`](https://torch.mlverse.org/docs/reference/dataloader.html).
  `shuffle=TRUE` by default for the training data and `batch_size=32` by
  default. It will error if not `NULL` and `data` is already a
  dataloader.

## See also

Other training:
[`evaluate()`](https://mlverse.github.io/luz/reference/evaluate.md),
[`fit.luz_module_generator()`](https://mlverse.github.io/luz/reference/fit.luz_module_generator.md),
[`setup()`](https://mlverse.github.io/luz/reference/setup.md)
