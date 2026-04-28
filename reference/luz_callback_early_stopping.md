# Early stopping callback

Stops training when a monitored metric stops improving

## Usage

``` r
luz_callback_early_stopping(
  monitor = "valid_loss",
  min_delta = 0,
  patience = 0,
  mode = "min",
  baseline = NULL
)
```

## Arguments

- monitor:

  A string in the format `<set>_<metric>` where `<set>` can be 'train'
  or 'valid' and `<metric>` can be the abbreviation of any metric that
  you are tracking during training. The metric name is case insensitive.

- min_delta:

  Minimum improvement to reset the patience counter.

- patience:

  Number of epochs without improving until stoping training.

- mode:

  Specifies the direction that is considered an improvement. By default
  'min' is used. Can also be 'max' (higher is better) and 'zero' (closer
  to zero is better).

- baseline:

  An initial value that will be used as the best seen value in the
  begining. Model will stop training if no better than baseline value is
  found in the first `patience` epochs.

## Value

A `luz_callback` that does early stopping.

## Note

This callback adds a `on_early_stopping` callback that can be used to
call callbacks as soon as the model stops training.

If `verbose=TRUE` in
[`fit.luz_module_generator()`](https://mlverse.github.io/luz/reference/fit.luz_module_generator.md)
a message is printed when early stopping.

## See also

Other luz_callbacks:
[`luz_callback()`](https://mlverse.github.io/luz/reference/luz_callback.md),
[`luz_callback_auto_resume()`](https://mlverse.github.io/luz/reference/luz_callback_auto_resume.md),
[`luz_callback_csv_logger()`](https://mlverse.github.io/luz/reference/luz_callback_csv_logger.md),
[`luz_callback_interrupt()`](https://mlverse.github.io/luz/reference/luz_callback_interrupt.md),
[`luz_callback_keep_best_model()`](https://mlverse.github.io/luz/reference/luz_callback_keep_best_model.md),
[`luz_callback_lr_scheduler()`](https://mlverse.github.io/luz/reference/luz_callback_lr_scheduler.md),
[`luz_callback_metrics()`](https://mlverse.github.io/luz/reference/luz_callback_metrics.md),
[`luz_callback_mixed_precision()`](https://mlverse.github.io/luz/reference/luz_callback_mixed_precision.md),
[`luz_callback_mixup()`](https://mlverse.github.io/luz/reference/luz_callback_mixup.md),
[`luz_callback_model_checkpoint()`](https://mlverse.github.io/luz/reference/luz_callback_model_checkpoint.md),
[`luz_callback_profile()`](https://mlverse.github.io/luz/reference/luz_callback_profile.md),
[`luz_callback_progress()`](https://mlverse.github.io/luz/reference/luz_callback_progress.md),
[`luz_callback_resume_from_checkpoint()`](https://mlverse.github.io/luz/reference/luz_callback_resume_from_checkpoint.md),
[`luz_callback_train_valid()`](https://mlverse.github.io/luz/reference/luz_callback_train_valid.md)

## Examples

``` r
cb <- luz_callback_early_stopping()
```
