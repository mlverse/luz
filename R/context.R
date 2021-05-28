#' Context object
#'
#' Context objects used in luz to share information between model methods,
#' metrics and callbacks.
#'
#' @includeRmd man/rmd/ctx.Rmd details
#' @rdname ctx
#' @name ctx
#'
#' @seealso Context object: [context]
NULL

#' Context object
#'
#' @description
#' Context object storing information about the model training context.
#' See also [ctx].
#'
#' @param name name of the metric
#' @param value value to log
#' @param what (string) What you are logging.
#' @param set (string) Usually 'train' or 'valid' indicating the set you want
#'  to lot to. But can be arbitrary info.
#' @param value Arbitrary value to log.
#' @param index Index that this value should be logged. If `NULL` the value
#'  is added to the end of list, otherwise the index is used.
#' @param append If `TRUE` and a value in the corresponding index already
#'  exists, then value is appended to the current value. If `FALSE` value
#'  is overwritten in favor of the new value.
#' @param epoch The epoch you want to extract metrics from.
#'
context <- R6::R6Class(
  "luz_context",
  lock_objects = FALSE,
  public = list(
    #' @description
    #' Allows logging arbitrary information in the `ctx`.
    log = function(what, set, value, index = NULL, append = TRUE) {

      if (is.null(index)) {
        index <- length(private$.records[[what]][[set]]) + 1L
      }

      current <- if (append) {
        if (length(private$.records[[what]][[set]]) < index) {
          NULL
        } else {
          private$.records[[what]][[set]][[index]]
        }
      } else {
        NULL
      }

      value <- append(current, value)

      if (is.null(private$.records[[what]]))
        private$.records[[what]][[set]] <- list()

      private$.records[[what]][[set]][[index]] <- value
      invisible(self)
    },
    #' @description
    #' Log a metric gen its name and value.
    #' Metric values are indexed by epoch.
    log_metric = function(name, value) {
      set <- if (self$training) "train" else "valid"

      value <- list(value)
      names(value) <- name

      self$log("metrics", set, value, index = self$epoch)


      invisible(self)
    },
    #' @description
    #' Get an specific value from the log.
    get_log = function(what, set, index = NULL) {
      if (is.null(index)) {
        index <- length(private$.records[[what]][[set]])
      }

      val <- private$.records[[what]][[set]]

      if (length(val) < index)
        return(NULL)

      val[[index]]
    },
    #' @description
    #' Get all metric given an epoch and set.
    get_metrics = function(set, epoch) {
      if (is.null(epoch)) {
        epoch <- length(private$.records[[what]][[set]])
      }
      self$get_log("metrics", set, epoch)
    },
    #' @description
    #' Get the value of a metric given its name, epoch and set.
    get_metric = function(name, set, epoch= NULL) {
      self$get_metrics(set, epoch)[[name]]
    },
    #' @description Allows setting the `verbose` attribute.
    #' @param verbose boolean. If `TRUE` verbose mode is used. If `FALSE` non verbose.
    #'   if `NULL` we use the result of [interactive()].
    set_verbose = function(verbose = NULL) {
      if (is.null(verbose)) {
        self$verbose <- interactive()
      } else {
        self$verbose <- verbose
      }
    }
  ),
  active = list(
    #' @field records stores information about values logged with `self$log`.
    records = function(x) {
      if (!missing(x))
        rlang::abort("Not allowed to modify records manually. Use ctx$log() or ctx$log_metric()")

      private$.records
    },
    #' @field device allows querying the current accelerator device
    device = function(x) {

      if (!missing(x))
        rlang::abort("Not allowed to modify the device manually. Modify the ctx$accelerator")

      if (is.null(self$accelerator))
        rlang::abort("Context doesn't have an accelerator attached.")

      self$accelerator$device
    }
  ),
  private = list(
    .records = list(metrics = list(
      train = list(),
      valid = list()
    ))
  )
)
