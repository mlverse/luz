# Resume training callback

This callback allows you to resume training a model.

## Usage

``` r
luz_callback_auto_resume(path = "./state.pt")
```

## Arguments

- path:

  Path to save state files for the model.

## Details

When using it, model weights, optimizer state are serialized at the end
of each epoch. If something fails during training simply re-running the
same script will restart the model training from the epoch right after
the last epoch that was serialized.

## Note

In general you will want to add this callback as the last in the
callbacks list, this way, the serialized state is likely to contain all
possible changes that other callbacks could have made at
`'on_epoch_end'`. The default `weight` attribute of this callback is
`Inf`.

Read the checkpointing article in the pkgdown website for more
information.

## Customizing serialization

By default model, optimizer state and records are serialized. Callbacks
can be used to customize serialization by implementing the
`state_dict()` and
[`load_state_dict()`](https://torch.mlverse.org/docs/reference/load_state_dict.html)
methods. If those methods are implemented, then `state_dict()` is called
at the end of each epoch and
[`load_state_dict()`](https://torch.mlverse.org/docs/reference/load_state_dict.html)
is called when the model is resumed.

## See also

Other luz_callbacks:
[`luz_callback()`](https://mlverse.github.io/luz/reference/luz_callback.md),
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
[`luz_callback_resume_from_checkpoint()`](https://mlverse.github.io/luz/reference/luz_callback_resume_from_checkpoint.md),
[`luz_callback_train_valid()`](https://mlverse.github.io/luz/reference/luz_callback_train_valid.md)

## Examples

``` r
if (torch::torch_is_installed()) {
library(torch)
library(luz)

x <- torch_randn(1000, 10)
y <- torch_randn(1000, 1)

model <- nn_linear %>%
  setup(optimizer = optim_sgd, loss = nnf_mse_loss) %>%
  set_hparams(in_features = 10, out_features = 1) %>%
  set_opt_hparams(lr = 0.01)


# simulate a failure in the middle of epoch 5 happening only once.
callback_stop <- luz_callback(
  "interrupt",
  failed = FALSE,
  on_epoch_end = function() {
    if (ctx$epoch == 5 && !self$failed) {
      self$failed <- TRUE
      stop("Error on epoch 5")
    }
  }
)

path <- tempfile()
autoresume <- luz_callback_auto_resume(path = path)
interrupt <- callback_stop()

# try once and the model fails
try({
  results <- model %>% fit(
    list(x, y),
    callbacks = list(autoresume, interrupt),
    verbose = FALSE
  )
})

# model resumes and completes
results <- model %>% fit(
  list(x, y),
  callbacks = list(autoresume, interrupt),
  verbose = FALSE
)

get_metrics(results)

}
#> Error in FUN(X[[i]], ...) : 
#>   Error while calling callback with class <interrupt/LuzCallback/R6> at
#> on_epoch_end.
#> Caused by error in `self[[callback_nm]]()`:
#> ! Error on epoch 5
#>      set metric epoch    value
#> 1  train   loss     1 1.243438
#> 2  train   loss     2 1.079375
#> 3  train   loss     3 1.030074
#> 4  train   loss     4 1.015318
#> 5  train   loss     5 1.016665
#> 6  train   loss     6 1.014279
#> 7  train   loss     7 1.012111
#> 8  train   loss     8 1.003616
#> 9  train   loss     9 1.003095
#> 10 train   loss    10 1.009394
```
