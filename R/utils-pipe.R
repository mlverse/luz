#' Pipe operator
#'
#' See \code{magrittr::\link[magrittr:pipe]{\%>\%}} for details.
#'
#' @name %>%
#' @rdname pipe
#' @keywords internal
#' @export
#' @importFrom magrittr %>%
#' @importFrom zeallot %<-%
#'
#' @usage lhs \%>\% rhs
NULL

`%||%` <- function(x, y) {
  if (is.null(x))
    y
  else
    x
}

utils::globalVariables(c("self"))

has_method <- function(x, name) {
  if (!is.null(x$public_methods[[name]]))
    TRUE
  else if (!is.null(x$get_inherit()))
    has_method(x$get_inherit(), name)
  else
    FALSE
}


get_forward <- function(x) {
  if (!is.null(x$public_methods[["forward"]]))
    x$public_methods[["forward"]]
  else if (!is.null(x$get_inherit()))
    get_forward(x$get_inherit())
  else
    rlang::abort("No `forward` method found.")
}

has_forward_method <- function(x) {
  test_module <- torch::nn_module(initialize = function() {})
  nn_forward <- test_module$get_inherit()$public_methods$forward
  forward <- get_forward(x)
  !isTRUE(identical(nn_forward, forward))
}

bind_context <- function(x, ctx) {
  e <- rlang::fn_env(x$clone) # the `clone` method must always exist in R6 classes
  rlang::env_bind(e, ctx = ctx)
}

test <- R6::R6Class(
  "help",
  public = list(
    x = NULL,
    initialize = function(x) {
      self$x = x
    },
    find_ctx = function() {
      ctx
    }
  )
)
