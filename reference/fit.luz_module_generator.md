# Fit a `nn_module`

Fit a `nn_module`

## Usage

``` r
# S3 method for class 'luz_module_generator'
fit(
  object,
  data,
  epochs = 10,
  callbacks = NULL,
  valid_data = NULL,
  accelerator = NULL,
  verbose = NULL,
  ...,
  dataloader_options = NULL
)
```

## Arguments

- object:

  An `nn_module` that has been
  [`setup()`](https://mlverse.github.io/luz/reference/setup.md).

- data:

  (dataloader, dataset or list) A dataloader created with
  [`torch::dataloader()`](https://torch.mlverse.org/docs/reference/dataloader.html)
  used for training the model, or a dataset created with
  [`torch::dataset()`](https://torch.mlverse.org/docs/reference/dataset.html)
  or a list. Dataloaders and datasets must return a list with at most 2
  items. The first item will be used as input for the module and the
  second will be used as a target for the loss function.

- epochs:

  (int) The maximum number of epochs for training the model. If a single
  value is provided, this is taken to be the `max_epochs` and
  `min_epochs` is set to 0. If a vector of two numbers is provided, the
  first value is `min_epochs` and the second value is `max_epochs`. The
  minimum and maximum number of epochs are included in the context
  object as `ctx$min_epochs` and `ctx$max_epochs`, respectively.

- callbacks:

  (list, optional) A list of callbacks defined with
  [`luz_callback()`](https://mlverse.github.io/luz/reference/luz_callback.md)
  that will be called during the training procedure. The callbacks
  [`luz_callback_metrics()`](https://mlverse.github.io/luz/reference/luz_callback_metrics.md),
  [`luz_callback_progress()`](https://mlverse.github.io/luz/reference/luz_callback_progress.md)
  and
  [`luz_callback_train_valid()`](https://mlverse.github.io/luz/reference/luz_callback_train_valid.md)
  are always added by default.

- valid_data:

  (dataloader, dataset, list or scalar value; optional) A dataloader
  created with
  [`torch::dataloader()`](https://torch.mlverse.org/docs/reference/dataloader.html)
  or a dataset created with
  [`torch::dataset()`](https://torch.mlverse.org/docs/reference/dataset.html)
  that will be used during the validation procedure. They must return a
  list with (input, target). If `data` is a torch dataset or a list,
  then you can also supply a numeric value between 0 and 1 - and in this
  case a random sample with size corresponding to that proportion from
  `data` will be used for validation.

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

- ...:

  Currently unused.

- dataloader_options:

  Options used when creating a dataloader. See
  [`torch::dataloader()`](https://torch.mlverse.org/docs/reference/dataloader.html).
  `shuffle=TRUE` by default for the training data and `batch_size=32` by
  default. It will error if not `NULL` and `data` is already a
  dataloader.

## Value

A fitted object that can be saved with
[`luz_save()`](https://mlverse.github.io/luz/reference/luz_save.md) and
can be printed with [`print()`](https://rdrr.io/r/base/print.html) and
plotted with [`plot()`](https://rdrr.io/r/graphics/plot.default.html).

## See also

[`predict.luz_module_fitted()`](https://mlverse.github.io/luz/reference/predict.luz_module_fitted.md)
for how to create predictions.
[`setup()`](https://mlverse.github.io/luz/reference/setup.md) to find
out how to create modules that can be trained with `fit`.

Other training:
[`evaluate()`](https://mlverse.github.io/luz/reference/evaluate.md),
[`predict.luz_module_fitted()`](https://mlverse.github.io/luz/reference/predict.luz_module_fitted.md),
[`setup()`](https://mlverse.github.io/luz/reference/setup.md)
