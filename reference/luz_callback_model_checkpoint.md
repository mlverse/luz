# Checkpoints model weights

This saves checkpoints of the model according to the specified metric
and behavior.

## Usage

``` r
luz_callback_model_checkpoint(
  path,
  monitor = "valid_loss",
  save_best_only = FALSE,
  mode = "min",
  min_delta = 0
)
```

## Arguments

- path:

  Path to save the model on disk. The path is interpolated with `glue`,
  so you can use any attribute within the
  [ctx](https://mlverse.github.io/luz/reference/ctx.md) by using
  `'{ctx$epoch}'`. Specially the `epoch` and `monitor` quantities are
  already in the environment. If the specified path is a path to a
  directory (ends with `/` or `\`), then models are saved with the name
  given by `epoch-{epoch:02d}-{self$monitor}-{monitor:.3f}.pt`. See more
  in the examples. You can use
  [`sprintf()`](https://rdrr.io/r/base/sprintf.html) to quickly format
  quantities, for example:`'{epoch:02d}'`.

- monitor:

  A string in the format `<set>_<metric>` where `<set>` can be 'train'
  or 'valid' and `<metric>` can be the abbreviation of any metric that
  you are tracking during training. The metric name is case insensitive.

- save_best_only:

  if `TRUE` models are only saved if they have an improvement over a
  previously saved model.

- mode:

  Specifies the direction that is considered an improvement. By default
  'min' is used. Can also be 'max' (higher is better) and 'zero' (closer
  to zero is better).

- min_delta:

  Minimum difference to consider as improvement. Only used when
  `save_best_only=TRUE`.

## Note

`mode` and `min_delta` are only used when `save_best_only=TRUE`.
`save_best_only` will overwrite the saved models if the `path` parameter
don't differentiate by epochs.

Read the checkpointing article in the pkgdown website for more
information.

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
[`luz_callback_profile()`](https://mlverse.github.io/luz/reference/luz_callback_profile.md),
[`luz_callback_progress()`](https://mlverse.github.io/luz/reference/luz_callback_progress.md),
[`luz_callback_resume_from_checkpoint()`](https://mlverse.github.io/luz/reference/luz_callback_resume_from_checkpoint.md),
[`luz_callback_train_valid()`](https://mlverse.github.io/luz/reference/luz_callback_train_valid.md)

## Examples

``` r
luz_callback_model_checkpoint(path= "path/to/dir")
#> <model_checkpoint_callback>
#>   Inherits from: <monitor_metrics>
#>   Public:
#>     call: function (callback_nm) 
#>     clone: function (deep = FALSE) 
#>     compare: function (new, old) 
#>     find_quantity: function () 
#>     fmt_path: function (path) 
#>     initialize: function (path, monitor = "valid_loss", save_best_only = FALSE, 
#>     min_delta: 0
#>     mode: min
#>     monitor: valid_loss
#>     on_epoch_end: function () 
#>     path: path/to/dir
#>     save_best_only: FALSE
#>     set_ctx: function (ctx) 
luz_callback_model_checkpoint(path= "path/to/dir/epoch-{epoch:02d}/model.pt")
#> <model_checkpoint_callback>
#>   Inherits from: <monitor_metrics>
#>   Public:
#>     call: function (callback_nm) 
#>     clone: function (deep = FALSE) 
#>     compare: function (new, old) 
#>     find_quantity: function () 
#>     fmt_path: function (path) 
#>     initialize: function (path, monitor = "valid_loss", save_best_only = FALSE, 
#>     min_delta: 0
#>     mode: min
#>     monitor: valid_loss
#>     on_epoch_end: function () 
#>     path: path/to/dir/epoch-{epoch:02d}/model.pt
#>     save_best_only: FALSE
#>     set_ctx: function (ctx) 
luz_callback_model_checkpoint(path= "path/to/dir/epoch-{epoch:02d}/model-{monitor:.2f}.pt")
#> <model_checkpoint_callback>
#>   Inherits from: <monitor_metrics>
#>   Public:
#>     call: function (callback_nm) 
#>     clone: function (deep = FALSE) 
#>     compare: function (new, old) 
#>     find_quantity: function () 
#>     fmt_path: function (path) 
#>     initialize: function (path, monitor = "valid_loss", save_best_only = FALSE, 
#>     min_delta: 0
#>     mode: min
#>     monitor: valid_loss
#>     on_epoch_end: function () 
#>     path: path/to/dir/epoch-{epoch:02d}/model-{monitor:.2f}.pt
#>     save_best_only: FALSE
#>     set_ctx: function (ctx) 
```
