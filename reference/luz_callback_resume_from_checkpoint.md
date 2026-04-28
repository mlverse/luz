# Allow resume model training from a specific checkpoint

Allow resume model training from a specific checkpoint

## Usage

``` r
luz_callback_resume_from_checkpoint(
  path,
  ...,
  restore_model_state = TRUE,
  restore_records = FALSE,
  restore_optimizer_state = FALSE,
  restore_callbacks_state = FALSE
)
```

## Arguments

- path:

  Path to the checkpoint that you want to resume.

- ...:

  currently unused.

- restore_model_state:

  Wether to restore the model state from the checkpoint.

- restore_records:

  Wether to restore records from the checkpoint.

- restore_optimizer_state:

  Wether to restore the optimizer state from the checkpoint.

- restore_callbacks_state:

  Wether to restore the callbacks state from the checkpoint.

## Note

Read the checkpointing article in the pkgdown website for more
information.

## See also

[`luz_callback_model_checkpoint()`](https://mlverse.github.io/luz/reference/luz_callback_model_checkpoint.md)

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
[`luz_callback_progress()`](https://mlverse.github.io/luz/reference/luz_callback_progress.md),
[`luz_callback_train_valid()`](https://mlverse.github.io/luz/reference/luz_callback_train_valid.md)
