#' Creates a dataloader from its input
#'
#' `as_dataloader` is used internally by luz to convert input
#' `data` and `valid_data` as passed to [fit.luz_module_generator()] to a
#' [torch::dataloader]
#'
#' `as_dataloader` methods should have sensible defaults for batch_size,
#' parallel workers, etc.
#'
#' It allows users to quickly experiment with [fit.luz_module_generator()] by not requiring
#' to create a [torch::dataset] and a [torch::dataloader] in simple
#' experiments.
#'
#' @section Overriding:
#'
#' You can implement your own `as_dataloader` S3 method if you want your data
#' structure to be automatically supported by luz's [fit.luz_module_generator()].
#' The method must satisfy the following conditions:
#'
#' - The method should return a [torch::dataloader()].
#' - The only required argument is `x`. You have good default for all other
#'   arguments.
#'
#' It's better to avoid implementing `as_dataloader` methods for common S3 classes
#' like `data.frames`. In this case, its better to assign a different class to
#' the inputs and implement `as_dataloader` for it.
#'
#' @param x the input object.
#' @param ... used by the specialized methods.
#'
#' @export
as_dataloader <- function(x, ...) {
  UseMethod("as_dataloader")
}

#' @export
as_dataloader.default <- function(x, ...) {
  if (is.null(x))
    return(x)

  rlang::abort(paste0(
    "Unsupported object with class '",
    class(x)[1],
    "'."
  ), class = "value_error")
}

#' @inheritParams torch::dataloader
#' @param ... Passed to [torch::dataloader()].
#' @describeIn as_dataloader Converts a [torch::dataset()] to a [torch::dataloader()].
#' @export
as_dataloader.dataset <- function(x, ..., batch_size = 32) {
  torch::dataloader(dataset = x, batch_size = batch_size, ...)
}

#' @describeIn as_dataloader Converts a list of tensors or arrays with the same
#'   size in the first dimension to a  [torch::dataloader()]
#' @export
as_dataloader.list <- function(x, ...) {
  dataset <- as_dataset(x)
  as_dataloader(dataset, ...)
}

#' @describeIn as_dataloader Returns the same dataloader
#' @export
as_dataloader.dataloader <- function(x, ...) {
  x
}

#' @describeIn as_dataloader Converts the matrix to a dataloader
#' @export
as_dataloader.matrix <- function(x, ...) {
  as_dataloader(list(x))
}

#' @describeIn as_dataloader Converts the numeric vector to a dataloader
#' @export
as_dataloader.numeric <- as_dataloader.matrix

#' @describeIn as_dataloader Converts the array to a dataloader
#' @export
as_dataloader.array <- as_dataloader.matrix

#' @describeIn as_dataloader Converts the tensor to a dataloader
#' @export
as_dataloader.torch_tensor <- as_dataloader.matrix

as_dataset <- function(x, ...) {
  UseMethod("as_dataset")
}

#' @export
as_dataset.default <- function(x, ...) {
  rlang::abort(sprinf("Can't convert object with class '%s' to a torch dataset.", class(x)[1]))
}

#' @export
as_dataset.dataset <- function(x, ...) {
  x
}

#' @export
as_dataset.list <- function(x, ...) {
  tensors <- lapply(x, function(x) {
    if (inherits(x, "torch_tensor"))
      x
    else
      torch::torch_tensor(x)
  })
  do.call(torch::tensor_dataset, tensors)
}

#' @export
as_dataset.matrix <- function(x, ...) {
  as_dataset(list(x))
}

#' @export
as_dataset.numeric <- as_dataset.matrix

#' @export
as_dataset.array <- as_dataset.matrix

#' @export
as_dataset.torch_tensor <- as_dataset.matrix

