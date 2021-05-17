#' Context object
#'
#' Context objects used in luz to share information between model methods,
#' metrics and callbacks.
#'
#' @name ctx
#'
#' @includeRmd man/rmd/ctx.Rmd details
#' @rdname ctx
NULL

context <- R6::R6Class(
  "luz_context",
  lock_objects = FALSE,
  public = list(
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
    log_metric = function(name, value) {
      set <- if (self$training) "train" else "valid"

      value <- list(value)
      names(value) <- name

      self$log("metrics", set, value, index = self$epoch)


      invisible(self)
    },
    get_log = function(what, set, index = NULL) {
      if (is.null(index)) {
        index <- length(private$.records[[what]][[set]])
      }

      val <- private$.records[[what]][[set]]

      if (length(val) < index)
        return(NULL)

      val[[index]]
    },
    get_metrics = function(set, epoch) {
      if (is.null(epoch)) {
        epoch <- length(private$.records[[what]][[set]])
      }
      self$get_log("metrics", set, epoch)
    },
    get_metric = function(name, set, epoch= NULL) {
      self$get_metrics(set, epoch)[[name]]
    }
  ),
  active = list(
    records = function(x) {
      if (!missing(x))
        rlang::abort("Not allowed to modify records manually. Use ctx$log() or ctx$log_metric()")

      private$.records
    }
  ),
  private = list(
    .records = list(metrics = list(
      train = list(),
      valid = list()
    ))
  )
)
