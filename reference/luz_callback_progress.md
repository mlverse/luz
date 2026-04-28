# Progress callback

Responsible for printing progress during training.

## Usage

``` r
luz_callback_progress()
```

## Value

A `luz_callback`

## Note

In general you don't need to use these callback by yourself because it's
always included by default in
[`fit.luz_module_generator()`](https://mlverse.github.io/luz/reference/fit.luz_module_generator.md).

Printing can be disabled by passing `verbose=FALSE` to
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
[`luz_callback_metrics()`](https://mlverse.github.io/luz/reference/luz_callback_metrics.md),
[`luz_callback_mixed_precision()`](https://mlverse.github.io/luz/reference/luz_callback_mixed_precision.md),
[`luz_callback_mixup()`](https://mlverse.github.io/luz/reference/luz_callback_mixup.md),
[`luz_callback_model_checkpoint()`](https://mlverse.github.io/luz/reference/luz_callback_model_checkpoint.md),
[`luz_callback_profile()`](https://mlverse.github.io/luz/reference/luz_callback_profile.md),
[`luz_callback_resume_from_checkpoint()`](https://mlverse.github.io/luz/reference/luz_callback_resume_from_checkpoint.md),
[`luz_callback_train_valid()`](https://mlverse.github.io/luz/reference/luz_callback_train_valid.md)
