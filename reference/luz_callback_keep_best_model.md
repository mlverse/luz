# Keep the best model

Each epoch, if there's improvement in the monitored metric we serialize
the model weights to a temp file. When training is done, we reload
weights from the best model.

## Usage

``` r
luz_callback_keep_best_model(
  monitor = "valid_loss",
  mode = "min",
  min_delta = 0
)
```

## Arguments

- monitor:

  A string in the format `<set>_<metric>` where `<set>` can be 'train'
  or 'valid' and `<metric>` can be the abbreviation of any metric that
  you are tracking during training. The metric name is case insensitive.

- mode:

  Specifies the direction that is considered an improvement. By default
  'min' is used. Can also be 'max' (higher is better) and 'zero' (closer
  to zero is better).

- min_delta:

  Minimum improvement to reset the patience counter.

## See also

Other luz_callbacks:
[`luz_callback()`](https://mlverse.github.io/luz/reference/luz_callback.md),
[`luz_callback_auto_resume()`](https://mlverse.github.io/luz/reference/luz_callback_auto_resume.md),
[`luz_callback_csv_logger()`](https://mlverse.github.io/luz/reference/luz_callback_csv_logger.md),
[`luz_callback_early_stopping()`](https://mlverse.github.io/luz/reference/luz_callback_early_stopping.md),
[`luz_callback_interrupt()`](https://mlverse.github.io/luz/reference/luz_callback_interrupt.md),
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
cb <- luz_callback_keep_best_model()
```
