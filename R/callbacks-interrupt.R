#' @include callbacks.R
NULL

#' Interrupt callback
#'
#' Adds a handler that allows interrupting the training loop using `ctrl + C`.
#' Also registers a `on_interrupt` breakpoint so users can register callbacks to
#' be run on training loop interruption.
#'
#' @note In general you don't need to use these callback by yourself because it's always
#'   included by default in [fit.luz_module_generator()].
#'
#' @family luz_callbacks
#' @export
luz_callback_interrupt <- luz_callback(
  "interrupt_callback",
  on_fit_begin = function() {
    ctx$handlers <- append(ctx$handlers, list(
      interrupt = function(err) {
        ctx$call_callbacks("on_interrupt")
        invisible(NULL)
      }
    ))
  }
)
