# Evaluates a fitted model on a dataset

Evaluates a fitted model on a dataset

## Usage

``` r
evaluate(
  object,
  data,
  ...,
  metrics = NULL,
  callbacks = list(),
  accelerator = NULL,
  verbose = NULL,
  dataloader_options = NULL
)
```

## Arguments

- object:

  A fitted model to evaluate.

- data:

  (dataloader, dataset or list) A dataloader created with
  [`torch::dataloader()`](https://torch.mlverse.org/docs/reference/dataloader.html)
  used for training the model, or a dataset created with
  [`torch::dataset()`](https://torch.mlverse.org/docs/reference/dataset.html)
  or a list. Dataloaders and datasets must return a list with at most 2
  items. The first item will be used as input for the module and the
  second will be used as a target for the loss function.

- ...:

  Currently unused.

- metrics:

  A list of luz metrics to be tracked during evaluation. If `NULL`
  (default) then the same metrics that were used during training are
  tracked.

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

## Details

Once a model has been trained you might want to evaluate its performance
on a different dataset. For that reason, luz provides the `?evaluate`
function that takes a fitted model and a dataset and computes the
metrics attached to the model.

Evaluate returns a `luz_module_evaluation` object that you can query for
metrics using the `get_metrics` function or simply `print` to see the
results.

For example:

    evaluation <- fitted %>% evaluate(data = valid_dl)
    metrics <- get_metrics(evaluation)
    print(evaluation)

    ## A `luz_module_evaluation`
    ## -- Results ---------------------------------------------------------------------
    ## loss: 1.8892
    ## mae: 1.0522
    ## mse: 1.645
    ## rmse: 1.2826

## See also

Other training:
[`fit.luz_module_generator()`](https://mlverse.github.io/luz/reference/fit.luz_module_generator.md),
[`predict.luz_module_fitted()`](https://mlverse.github.io/luz/reference/predict.luz_module_fitted.md),
[`setup()`](https://mlverse.github.io/luz/reference/setup.md)
