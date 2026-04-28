# Saves luz objects to disk

Allows saving luz fitted models to the disk. Objects can be loaded back
with
[`luz_load()`](https://mlverse.github.io/luz/reference/luz_load.md).

## Usage

``` r
luz_save(obj, path, ...)
```

## Arguments

- obj:

  an object of class 'luz_module_fitted' as returned by
  [`fit.luz_module_generator()`](https://mlverse.github.io/luz/reference/fit.luz_module_generator.md).

- path:

  path in file system to the object.

- ...:

  currently unused.

## Note

Objects are saved as plain `.rds` files but `obj$model` is serialized
with `torch_save` before saving it.

## Warning

The [ctx](https://mlverse.github.io/luz/reference/ctx.md) is naively
serialized. Ie, we only use
[`saveRDS()`](https://rdrr.io/r/base/readRDS.html) to serialize it.
Don't expect `luz_save` to work correctly if you have unserializable
objects in the [ctx](https://mlverse.github.io/luz/reference/ctx.md)
like `torch_tensor`s and external pointers in general.

## See also

Other luz_save:
[`luz_load()`](https://mlverse.github.io/luz/reference/luz_load.md)
