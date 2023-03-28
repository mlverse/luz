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

get_method <- function(x, method) {
  if (!is.null(x$public_methods[[method]]))
    x$public_methods[[method]]
  else if (!is.null(x$get_inherit()))
    get_method(x$get_inherit(), method)
  else
    NULL
}

get_forward <- function(x) {
  forward <- get_method(x, "forward")
  if (is.null(forward)) {
    cli::cli_abort("No method {.val forward} method found.")
  }
  forward
}

has_forward_method <- function(x) {
  test_module <- torch::nn_module(initialize = function() {})
  nn_forward <- test_module$get_inherit()$public_methods$forward
  forward <- get_forward(x)
  !isTRUE(identical(nn_forward, forward))
}

bind_context <- function(x, ctx) {
  e <- x$.__enclos_env__
  rlang::env_bind(e, ctx = ctx)

  if (!is.null(x <- x$.__enclos_env__$super))
    bind_context(x, ctx)

  invisible(NULL)
}

get_init <- function(x) {

  if (!is.null(x$public_methods[["initialize"]]))
    return(x$public_methods[["initialize"]])
  else
    return(get_init(x$get_inherit()))

}

inform <- function(message) {
  e <- rlang::caller_env()
  ctx <- rlang::env_get(e, "ctx", inherit = TRUE)

  verbose <- ctx$verbose

  if (verbose)
    rlang::inform(message)

  invisible(NULL)
}

utils::globalVariables(c("super"))

make_class <- function(name, ..., private, active, inherit, parent_env, .init_fun,
                       .out_class = NULL) {
  public <- rlang::list2(...)

  e <- new.env(parent = parent_env)

  e$inherit <- inherit

  r6_class <- R6::R6Class(
    classname = name,
    inherit = inherit,
    public = public,
    private = private,
    active = active,
    parent_env = e,
    lock_objects = FALSE
  )

  init <- get_init(r6_class)

  f <- rlang::new_function(
    args = rlang::fn_fmls(init),
    body = rlang::expr({
      obj <- R6::R6Class(
        inherit = r6_class,
        public = list(
          initialize = function() {
            super$initialize(!!!rlang::fn_fmls_syms(init))
          }
        ),
        private = private,
        active = active,
        lock_objects = FALSE,
        parent_env = rlang::current_env()
      )
      if (.init_fun) {
        r6_class$new(!!!rlang::fn_fmls_syms(init))
      } else {
        if (is.null(.out_class)) stop("Should have an out class.")
        structure(list(
          new = function() r6_class$new(!!!rlang::fn_fmls_syms(init))
        ), class = .out_class)
      }
    })
  )
  attr(f, "r6_class") <- r6_class
  f
}

# from https://glue.tidyverse.org/articles/transformers.html
sprintf_transformer <- function(text, envir) {
  m <- regexpr(":.+$", text)
  if (m != -1) {
    format <- substring(regmatches(text, m), 2)
    regmatches(text, m) <- ""
    res <- eval(parse(text = text, keep.source = FALSE), envir)
    do.call(sprintf, list(glue::glue("%{format}"), res))
  } else {
    eval(parse(text = text, keep.source = FALSE), envir)
  }
}

check_installed <- function (pkg, fun) {
  if (rlang::is_installed(pkg)) {
    return()
  }
  rlang::abort(c(paste0("The ", pkg, " package must be installed in order to use `",
                 fun, "`"), i = paste0("Do you need to run `install.packages('",
                                       pkg, "')`?")))
}

map2 <- function(x, y, f) {
  if (length(x) != length(y)) rlang::abort("Objects must have the same length.")
  out <- vector(mode = "list", length = length(x))
  for(i in seq_along(x)) {
    out[[i]] <- f(x[[i]], y[[i]])
  }
  names(out) <- names(x)
  out
}

with_handlers <- function(..., .expr) {
  rlang::try_fetch(.expr, ...)
}
