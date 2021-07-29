#' @include callbacks.R
NULL

#' Validation Check
#'
#' Check validation loop before fitting model.
#'
#' @param batches Number of validation batches to check. Default is 2.
#'
#' @note Usually the training loop is much longer than the validation
#'   loop and issues with the validation loop aren't encountered until after
#'   a long training runtime. This callback runs the validation loop first on
#'   `batches` number of batches and then proceeds onto the standard
#'   training process.
#'
#' @note Printing can be disabled by passing `verbose = FALSE` to
#'   [fit.luz_module_generator()].
#'
#' @family luz_callbacks
#'
#' @returns
#' A `luz_callback`.
#'
#' @export
luz_callback_validation_check <- luz_callback(
  "validation_check_callback",
  initialize = function(batches = 2) {
    if (!rlang::is_scalar_integerish(batches)) {
      rlang::abort("`batches` must be a single integer value.")
    }
    self$batches <- batches
  },
  on_fit_begin = function() {
    if (is.null(ctx$valid_data)) return()
    if (self$batches <= 0) return()

    ctx$model$eval()
    ctx$training <- FALSE

    self$initialize_progress_bar()

    i <- 0
    torch::with_no_grad({
      coro::loop(for (batch in ctx$valid_data) {
        self$validate_one_batch(batch)
        self$tick_progress_bar(self$loss)
        i <- i + 1
        if (i >= self$batches) break()
      })
    })
  },
  validate_one_batch = function(batch) {
    input  <- list(batch[[1]])
    target <- batch[[2]]
    pred <- do.call(ctx$model, input)
    self$loss <- ctx$model$loss(pred, target)
  },
  initialize_progress_bar = function() {
    format <- "Validation check: :current/:total [:bar] - Loss: :loss"
    self$pb <- progress::progress_bar$new(
      force = getOption("luz.force_progress_bar", FALSE),
      show_after = 0,
      format = format,
      total = self$batches,
      clear = FALSE
    )
  },
  tick_progress_bar = function(token) {
    if (ctx$verbose) {
      loss <- format(round(as.numeric(token), digits = 4), nsmall = 4)
      self$pb$tick(tokens = list(loss = loss))
    }
  }
)
