# tfevents callback

Logs metrics and other model information in the tfevents file format.
Assuming tensorboard is installed, result can be visualized with

## Usage

``` r
luz_callback_tfevents(logdir = "logs", histograms = FALSE, ...)
```

## Arguments

- logdir:

  A directory to where log will be written to.

- histograms:

  A boolean specifying if histograms of model weights should be logged.
  It can also be a character vector specifying the name of the
  parameters that should be logged (names are the same as
  `names(model$parameters)`).

- ...:

  Currently not used. For future expansion.

## Details

    tensorboard --logdir=logs

## Examples

``` r
if (torch::torch_is_installed()) {
library(torch)
x <- torch_randn(1000, 10)
y <- torch_randn(1000, 1)

model <- nn_linear %>%
  setup(loss = nnf_mse_loss, optimizer = optim_adam) %>%
  set_hparams(in_features = 10, out_features = 1) %>%
  set_opt_hparams(lr = 1e-4)

tmp <- tempfile()

model %>% fit(list(x, y), valid_data = 0.2, callbacks = list(
  luz_callback_tfevents(tmp, histograms = TRUE)
))
}
#> A `luz_module_fitted`
#> ── Time ────────────────────────────────────────────────────────────────────────
#> • Total time: 1.5s
#> • Avg time per training epoch: 115ms
#> 
#> ── Results ─────────────────────────────────────────────────────────────────────
#> Metrics observed in the last epoch.
#> 
#> ℹ Training:
#> loss: 1.6208
#> 
#> ── Model ───────────────────────────────────────────────────────────────────────
#> An `nn_module` containing 11 parameters.
#> 
#> ── Parameters ──────────────────────────────────────────────────────────────────
#> • weight: Float [1:1, 1:10]
#> • bias: Float [1:1]
```
