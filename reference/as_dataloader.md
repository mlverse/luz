# Creates a dataloader from its input

`as_dataloader` is used internally by luz to convert input `data` and
`valid_data` as passed to
[`fit.luz_module_generator()`](https://mlverse.github.io/luz/reference/fit.luz_module_generator.md)
to a
[torch::dataloader](https://torch.mlverse.org/docs/reference/dataloader.html)

## Usage

``` r
as_dataloader(x, ...)

# S3 method for class 'dataset'
as_dataloader(x, ..., batch_size = 32)

# S3 method for class 'iterable_dataset'
as_dataloader(x, ..., batch_size = 32)

# S3 method for class 'list'
as_dataloader(x, ...)

# S3 method for class 'dataloader'
as_dataloader(x, ...)

# S3 method for class 'matrix'
as_dataloader(x, ...)

# S3 method for class 'numeric'
as_dataloader(x, ...)

# S3 method for class 'array'
as_dataloader(x, ...)

# S3 method for class 'torch_tensor'
as_dataloader(x, ...)
```

## Arguments

- x:

  the input object.

- ...:

  Passed to
  [`torch::dataloader()`](https://torch.mlverse.org/docs/reference/dataloader.html).

- batch_size:

  (int, optional): how many samples per batch to load (default: `1`).

## Details

`as_dataloader` methods should have sensible defaults for batch_size,
parallel workers, etc.

It allows users to quickly experiment with
[`fit.luz_module_generator()`](https://mlverse.github.io/luz/reference/fit.luz_module_generator.md)
by not requiring to create a
[torch::dataset](https://torch.mlverse.org/docs/reference/dataset.html)
and a
[torch::dataloader](https://torch.mlverse.org/docs/reference/dataloader.html)
in simple experiments.

## Methods (by class)

- `as_dataloader(dataset)`: Converts a
  [`torch::dataset()`](https://torch.mlverse.org/docs/reference/dataset.html)
  to a
  [`torch::dataloader()`](https://torch.mlverse.org/docs/reference/dataloader.html).

- `as_dataloader(iterable_dataset)`: Converts a
  [`torch::iterable_dataset()`](https://torch.mlverse.org/docs/reference/iterable_dataset.html)
  into a
  [`torch::dataloader()`](https://torch.mlverse.org/docs/reference/dataloader.html)

- `as_dataloader(list)`: Converts a list of tensors or arrays with the
  same size in the first dimension to a
  [`torch::dataloader()`](https://torch.mlverse.org/docs/reference/dataloader.html)

- `as_dataloader(dataloader)`: Returns the same dataloader

- `as_dataloader(matrix)`: Converts the matrix to a dataloader

- `as_dataloader(numeric)`: Converts the numeric vector to a dataloader

- `as_dataloader(array)`: Converts the array to a dataloader

- `as_dataloader(torch_tensor)`: Converts the tensor to a dataloader

## Overriding

You can implement your own `as_dataloader` S3 method if you want your
data structure to be automatically supported by luz's
[`fit.luz_module_generator()`](https://mlverse.github.io/luz/reference/fit.luz_module_generator.md).
The method must satisfy the following conditions:

- The method should return a
  [`torch::dataloader()`](https://torch.mlverse.org/docs/reference/dataloader.html).

- The only required argument is `x`. You have good default for all other
  arguments.

It's better to avoid implementing `as_dataloader` methods for common S3
classes like `data.frames`. In this case, its better to assign a
different class to the inputs and implement `as_dataloader` for it.
