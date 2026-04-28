# Gradient clipping callback

By adding the GradientClip callback, the gradient `norm_type`
(default:2) norm is clipped to at most `max_norm` (default:1) using
[`torch::nn_utils_clip_grad_norm_()`](https://torch.mlverse.org/docs/reference/nn_utils_clip_grad_norm_.html),
which can avoid loss divergence.

## Usage

``` r
luz_callback_gradient_clip(max_norm = 1, norm_type = 2)
```

## Arguments

- max_norm:

  (float or int): max norm of the gradients

- norm_type:

  (float or int): type of the used p-norm. Can be `Inf` for infinity
  norm.

## References

See FastAI
[documentation](https://docs.fast.ai/callback.training.html#GradientClip)
for the GradientClip callback.
