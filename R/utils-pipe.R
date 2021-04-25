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
