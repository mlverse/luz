# Context object

Context object storing information about the model training context. See
also [ctx](https://mlverse.github.io/luz/reference/ctx.md).

## Public fields

- `buffers`:

  This is a list of buffers that callbacks can use to write temporary
  information into `ctx`.

## Active bindings

- `records`:

  stores information about values logged with `self$log`.

- `device`:

  allows querying the current accelerator device

- `callbacks`:

  list of callbacks that will be called.

- `iter`:

  current iteration

- `batch`:

  the current batch data. a list with input data and targets.

- `input`:

  a shortcut for `ctx$batch[[1]]`

- `target`:

  a shortcut for `ctx$batch[[2]]`

- `min_epochs`:

  the minimum number of epochs that the model will run on.

- `max_epochs`:

  the maximum number of epochs that the model will run.

- `hparams`:

  a list of hyperparameters that were used to initialize `ctx$model`.

- `opt_hparams`:

  a list of hyperparameters used to initialize the `ctx$optimizers`.

- `train_data`:

  a dataloader that is used for training the model

- `valid_data`:

  a dataloader using during model validation

- `accelerator`:

  an
  [`accelerator()`](https://mlverse.github.io/luz/reference/accelerator.md)
  used to move data, model and etc the the correct device.

- `optimizers`:

  a named list of optimizers that will be used during model training.

- `verbose`:

  bool wether the process is in verbose mode or not.

- `handlers`:

  List of error handlers that can be used. See
  [`rlang::try_fetch()`](https://rlang.r-lib.org/reference/try_fetch.html)
  for more info.

- `epoch_handlers`:

  List of error handlers that can be used. See
  [`rlang::try_fetch()`](https://rlang.r-lib.org/reference/try_fetch.html)
  for more info.

- `training`:

  A bool indicating if the model is in training or validation mode.

- `model`:

  The model being trained.

- `pred`:

  Last predicted values.

- `opt`:

  Current optimizer.

- `opt_name`:

  Current optimizer name.

- `data`:

  Current dataloader in use.

- `loss_fn`:

  Loss function used to train the model

- `loss`:

  Last computed loss values. Detached from the graph.

- `loss_grad`:

  Last computed loss value, not detached, so you can do additional
  tranformation.

- `epoch`:

  Current epoch.

- `metrics`:

  List of metrics that are tracked by the process.

- `step_opt`:

  Defines how step is called for the optimizer. It must be a function
  taking an optimizer as argument.

## Methods

### Public methods

- [`context$new()`](#method-luz_context-new)

- [`context$log()`](#method-luz_context-log)

- [`context$log_metric()`](#method-luz_context-log_metric)

- [`context$get_log()`](#method-luz_context-get_log)

- [`context$get_metrics()`](#method-luz_context-get_metrics)

- [`context$get_metric()`](#method-luz_context-get_metric)

- [`context$get_formatted_metrics()`](#method-luz_context-get_formatted_metrics)

- [`context$get_metrics_df()`](#method-luz_context-get_metrics_df)

- [`context$set_verbose()`](#method-luz_context-set_verbose)

- [`context$clean()`](#method-luz_context-clean)

- [`context$call_callbacks()`](#method-luz_context-call_callbacks)

- [`context$state_dict()`](#method-luz_context-state_dict)

- [`context$unsafe_set_records()`](#method-luz_context-unsafe_set_records)

- [`context$clone()`](#method-luz_context-clone)

------------------------------------------------------------------------

### Method `new()`

Initializes the context object with minimal necessary information.

#### Usage

    context$new(verbose, accelerator, callbacks, training)

#### Arguments

- `verbose`:

  Whether the context should be in verbose mode or not.

- `accelerator`:

  A luz
  [`accelerator()`](https://mlverse.github.io/luz/reference/accelerator.md)
  that configures device placement and others.

- `callbacks`:

  A list of callbacks used by the model. See
  [`luz_callback()`](https://mlverse.github.io/luz/reference/luz_callback.md).

- `training`:

  A boolean that indicates if the context is in training mode or not.

------------------------------------------------------------------------

### Method [`log()`](https://rdrr.io/r/base/Log.html)

Allows logging arbitrary information in the `ctx`.

#### Usage

    context$log(what, set, value, index = NULL, append = TRUE)

#### Arguments

- `what`:

  (string) What you are logging.

- `set`:

  (string) Usually 'train' or 'valid' indicating the set you want to log
  to. But can be arbitrary info.

- `value`:

  Arbitrary value to log.

- `index`:

  Index that this value should be logged. If `NULL` the value is added
  to the end of list, otherwise the index is used.

- `append`:

  If `TRUE` and a value in the corresponding index already exists, then
  value is appended to the current value. If `FALSE` value is
  overwritten in favor of the new value.

------------------------------------------------------------------------

### Method `log_metric()`

Log a metric by its name and value. Metric values are indexed by epoch.

#### Usage

    context$log_metric(name, value)

#### Arguments

- `name`:

  name of the metric

- `value`:

  Arbitrary value to log.

------------------------------------------------------------------------

### Method `get_log()`

Get a specific value from the log.

#### Usage

    context$get_log(what, set, index = NULL)

#### Arguments

- `what`:

  (string) What you are logging.

- `set`:

  (string) Usually 'train' or 'valid' indicating the set you want to log
  to. But can be arbitrary info.

- `index`:

  Index that this value should be logged. If `NULL` the value is added
  to the end of list, otherwise the index is used.

------------------------------------------------------------------------

### Method [`get_metrics()`](https://mlverse.github.io/luz/reference/get_metrics.md)

Get all metric given an epoch and set.

#### Usage

    context$get_metrics(set, epoch = NULL)

#### Arguments

- `set`:

  (string) Usually 'train' or 'valid' indicating the set you want to log
  to. But can be arbitrary info.

- `epoch`:

  The epoch you want to extract metrics from.

------------------------------------------------------------------------

### Method `get_metric()`

Get the value of a metric given its name, epoch and set.

#### Usage

    context$get_metric(name, set, epoch = NULL)

#### Arguments

- `name`:

  name of the metric

- `set`:

  (string) Usually 'train' or 'valid' indicating the set you want to log
  to. But can be arbitrary info.

- `epoch`:

  The epoch you want to extract metrics from.

------------------------------------------------------------------------

### Method `get_formatted_metrics()`

Get formatted metrics values

#### Usage

    context$get_formatted_metrics(set, epoch = NULL)

#### Arguments

- `set`:

  (string) Usually 'train' or 'valid' indicating the set you want to log
  to. But can be arbitrary info.

- `epoch`:

  The epoch you want to extract metrics from.

------------------------------------------------------------------------

### Method `get_metrics_df()`

Get a data.frame containing all metrics.

#### Usage

    context$get_metrics_df()

------------------------------------------------------------------------

### Method `set_verbose()`

Allows setting the `verbose` attribute.

#### Usage

    context$set_verbose(verbose = NULL)

#### Arguments

- `verbose`:

  boolean. If `TRUE` verbose mode is used. If `FALSE` non verbose. if
  `NULL` we use the result of
  [`interactive()`](https://rdrr.io/r/base/interactive.html).

------------------------------------------------------------------------

### Method `clean()`

Removes unnecessary information from the context object.

#### Usage

    context$clean()

------------------------------------------------------------------------

### Method `call_callbacks()`

Call the selected callbacks. Where `name` is the callback types to call,
eg 'on_epoch_begin'.

#### Usage

    context$call_callbacks(name)

#### Arguments

- `name`:

  name of the metric

------------------------------------------------------------------------

### Method `state_dict()`

Returns a list containing minimal information from the context. Used to
create the returned values.

#### Usage

    context$state_dict()

------------------------------------------------------------------------

### Method `unsafe_set_records()`

Are you sure you know what you are doing?

#### Usage

    context$unsafe_set_records(records)

#### Arguments

- `records`:

  New set of records to be set.

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    context$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
