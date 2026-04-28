# Learning rate scheduler callback

Initializes and runs
[`torch::lr_scheduler()`](https://torch.mlverse.org/docs/reference/lr_scheduler.html)s.

## Usage

``` r
luz_callback_lr_scheduler(
  lr_scheduler,
  ...,
  call_on = "on_epoch_end",
  opt_name = NULL
)
```

## Arguments

- lr_scheduler:

  A
  [`torch::lr_scheduler()`](https://torch.mlverse.org/docs/reference/lr_scheduler.html)
  that will be initialized with the optimizer and the `...` parameters.

- ...:

  Additional arguments passed to `lr_scheduler` together with the
  optimizers.

- call_on:

  The callback breakpoint that `scheduler$step()` is called. Default is
  `'on_epoch_end'`. See
  [`luz_callback()`](https://mlverse.github.io/luz/reference/luz_callback.md)
  for more information.

- opt_name:

  name of the optimizer that will be affected by this callback. Should
  match the name given in `set_optimizers`. If your module has a single
  optimizer, `opt_name` is not used.

## Value

A
[`luz_callback()`](https://mlverse.github.io/luz/reference/luz_callback.md)
generator.

## See also

Other luz_callbacks:
[`luz_callback()`](https://mlverse.github.io/luz/reference/luz_callback.md),
[`luz_callback_auto_resume()`](https://mlverse.github.io/luz/reference/luz_callback_auto_resume.md),
[`luz_callback_csv_logger()`](https://mlverse.github.io/luz/reference/luz_callback_csv_logger.md),
[`luz_callback_early_stopping()`](https://mlverse.github.io/luz/reference/luz_callback_early_stopping.md),
[`luz_callback_interrupt()`](https://mlverse.github.io/luz/reference/luz_callback_interrupt.md),
[`luz_callback_keep_best_model()`](https://mlverse.github.io/luz/reference/luz_callback_keep_best_model.md),
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
if (torch::torch_is_installed()) {
cb <- luz_callback_lr_scheduler(torch::lr_step, step_size = 30)
}
```
