# Mixup callback

Implementation of ['mixup: Beyond Empirical Risk
Minimization'](https://arxiv.org/abs/1710.09412). As of today, tested
only for categorical data, where targets are expected to be integers,
not one-hot encoded vectors. This callback is supposed to be used
together with
[`nn_mixup_loss()`](https://mlverse.github.io/luz/reference/nn_mixup_loss.md).

## Usage

``` r
luz_callback_mixup(alpha = 0.4, ..., run_valid = FALSE, auto_loss = FALSE)
```

## Arguments

- alpha:

  parameter for the beta distribution used to sample mixing coefficients

- ...:

  currently unused. Just to force named arguments.

- run_valid:

  Should it run during validation

- auto_loss:

  Should it automatically modify the loss function? This will wrap the
  loss function to create the mixup loss. If `TRUE` make sure that your
  loss function does not apply reductions. If `run_valid=FALSE`, then
  loss will be mean reduced during validation.

## Value

A `luz_callback`

## Details

Overall, we follow the [fastai
implementation](https://github.com/fastai/fastai/blob/master/fastai/callback/mixup.py)
described
[here](https://forums.fast.ai/t/mixup-data-augmentation/22764). Namely,

- We work with a single dataloader only, randomly mixing two
  observations from the same batch.

- We linearly combine losses computed for both targets:
  `loss(output, new_target) = weight * loss(output, target1) + (1-weight) * loss(output, target2)`

- We draw different mixing coefficients for every pair.

- We replace `weight` with `weight = max(weight, 1-weight)` to avoid
  duplicates.

## See also

[`nn_mixup_loss()`](https://mlverse.github.io/luz/reference/nn_mixup_loss.md),
[`nnf_mixup()`](https://mlverse.github.io/luz/reference/nnf_mixup.md)

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
[`luz_callback_model_checkpoint()`](https://mlverse.github.io/luz/reference/luz_callback_model_checkpoint.md),
[`luz_callback_profile()`](https://mlverse.github.io/luz/reference/luz_callback_profile.md),
[`luz_callback_progress()`](https://mlverse.github.io/luz/reference/luz_callback_progress.md),
[`luz_callback_resume_from_checkpoint()`](https://mlverse.github.io/luz/reference/luz_callback_resume_from_checkpoint.md),
[`luz_callback_train_valid()`](https://mlverse.github.io/luz/reference/luz_callback_train_valid.md)

## Examples

``` r
if (torch::torch_is_installed()) {
mixup_callback <- luz_callback_mixup()
}
```
