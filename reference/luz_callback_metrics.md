# Metrics callback

Tracks metrics passed to
[`setup()`](https://mlverse.github.io/luz/reference/setup.md) during
training and validation.

## Usage

``` r
luz_callback_metrics()
```

## Value

A `luz_callback`

## Details

This callback takes care of 2
[ctx](https://mlverse.github.io/luz/reference/ctx.md) attributes:

- `ctx$metrics`: stores the current metrics objects that are initialized
  once for epoch, and are further
  [`update()`](https://rdrr.io/r/stats/update.html)d and `compute()`d
  every batch. You will rarely need to work with these metrics.

- `ctx$records$metrics`: Stores metrics per training/validation and
  epoch. The structure is very similar to `ctx$losses`.

## Note

In general you won't need to explicitly use the metrics callback as it's
used by default in
[`fit.luz_module_generator()`](https://mlverse.github.io/luz/reference/fit.luz_module_generator.md).

## See also

Other luz_callbacks:
[`luz_callback()`](https://mlverse.github.io/luz/reference/luz_callback.md),
[`luz_callback_auto_resume()`](https://mlverse.github.io/luz/reference/luz_callback_auto_resume.md),
[`luz_callback_csv_logger()`](https://mlverse.github.io/luz/reference/luz_callback_csv_logger.md),
[`luz_callback_early_stopping()`](https://mlverse.github.io/luz/reference/luz_callback_early_stopping.md),
[`luz_callback_interrupt()`](https://mlverse.github.io/luz/reference/luz_callback_interrupt.md),
[`luz_callback_keep_best_model()`](https://mlverse.github.io/luz/reference/luz_callback_keep_best_model.md),
[`luz_callback_lr_scheduler()`](https://mlverse.github.io/luz/reference/luz_callback_lr_scheduler.md),
[`luz_callback_mixed_precision()`](https://mlverse.github.io/luz/reference/luz_callback_mixed_precision.md),
[`luz_callback_mixup()`](https://mlverse.github.io/luz/reference/luz_callback_mixup.md),
[`luz_callback_model_checkpoint()`](https://mlverse.github.io/luz/reference/luz_callback_model_checkpoint.md),
[`luz_callback_profile()`](https://mlverse.github.io/luz/reference/luz_callback_profile.md),
[`luz_callback_progress()`](https://mlverse.github.io/luz/reference/luz_callback_progress.md),
[`luz_callback_resume_from_checkpoint()`](https://mlverse.github.io/luz/reference/luz_callback_resume_from_checkpoint.md),
[`luz_callback_train_valid()`](https://mlverse.github.io/luz/reference/luz_callback_train_valid.md)
