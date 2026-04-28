# Create an accelerator

Create an accelerator

## Usage

``` r
accelerator(
  device_placement = TRUE,
  cpu = FALSE,
  cuda_index = torch::cuda_current_device()
)
```

## Arguments

- device_placement:

  (logical) whether the `accelerator` object should handle device
  placement. Default: `TRUE`

- cpu:

  (logical) whether the training procedure should run on the CPU.

- cuda_index:

  (integer) index of the CUDA device to use if multiple GPUs are
  available. Default: the result of torch::cuda_current_device().
