# Loads model weights into a fitted object.

This can be useful when you have saved model checkpoints during training
and want to reload the best checkpoint in the end.

## Usage

``` r
luz_load_model_weights(obj, path, ...)

luz_save_model_weights(obj, path)
```

## Arguments

- obj:

  luz object to which you want to copy the new weights.

- path:

  path to saved model in disk.

- ...:

  other arguments passed to
  [`torch::torch_load()`](https://torch.mlverse.org/docs/reference/torch_load.html).

## Value

Returns `NULL` invisibly.

## Warning

`luz_save_model_weights` operates inplace, ie modifies the model object
to contain the new weights.
