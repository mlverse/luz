#' Creates a dataloader from its input
#'
#' `as_dataloader` is used internally by luz to convert input
#' `data` and `valid_data` as passed to [fit] to a [torch::dataloader]
#'
#' `as_dataloader` methods should have sensible defaults for batch_size,
#' parallel workers, etc.
#'
#' It allows users to quickly experiment with [fit] by not requiring
#' to create a [torch::dataset] and a [torch::dataloader] in simple
#' experiments.
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
  rlang::abort(paste0(
    "Unsupported object with class '",
    class(x)[1],
    "'."
  ), class = "value_error")
}


