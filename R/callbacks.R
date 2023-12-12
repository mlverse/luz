#' @include utils.R

LuzCallback <- R6::R6Class(
  "LuzCallback",
  lock_objects = FALSE,
  public = list(
    initialize = function() {},
    set_ctx = function(ctx) {
      self$ctx <- ctx
    },
    call = function(callback_nm) {
      if (is.null(self[[callback_nm]]))
        return(invisible())
      self[[callback_nm]]()
      invisible()
    }
  )
)

assert_is_callback <- function(cb) {
  if (!inherits(cb, "LuzCallback")) {
    message <- c(
      x = "Callbacks must have class {.cls LuzCallback} but got {.cls {class(cb)}}")
    if (rlang::is_function(cb)) {
      message <- c(
        message,
        i = "Perhaps you forgot to initialize the callback?"
      )
    }
    cli::cli_abort(message)
  }
  invisible(TRUE)
}

call_all_callbacks <- function(callbacks, name) {
  torch::local_no_grad()
  lapply(callbacks, function(callback) {
    rlang::try_fetch(
      callback$call(name),
      error = function(cnd) {
        cli::cli_abort(c(
          "Error while calling callback with class {.cls {class(callback)}} at {.field {name}}."
        ), parent = cnd)
      }
    )
  })
}

default_callbacks <- function() {
  list(
    luz_callback_profile(),
    luz_callback_train_valid(),
    luz_callback_metrics(),
    luz_callback_progress(),
    luz_callback_interrupt()
  )
}

default_predict_callbacks <- function() {
  list(
    luz_callback_progress(),
    luz_callback_interrupt()
  )
}

default_evaluate_callbacks <- function() {
  list(
    luz_callback_profile(),
    luz_callback_metrics(),
    luz_callback_progress(),
    luz_callback_interrupt()
  )
}

#' Create a new callback
#'
#' @param name name of the callback
#' @param ... Public methods of the callback. The name of the methods is used
#'  to know how they should be called. See the details section.
#' @inheritParams R6::R6Class
#'
#' @includeRmd man/rmd/callbacks.Rmd details
#'
#' @section Prediction callbacks:
#'
#' You can also use callbacks when using [predict()]. In this case the supported
#' callback methods are detailed above.
#'
#' ```
#' Start predict
#'  - on_predict_begin
#'  Start prediction loop
#'   - on_predict_batch_begin
#'   - on_predict_batch_end
#'  End prediction loop
#'  - on_predict_end
#' End predict
#' ```
#'
#' @section Evaluate callbacks:
#'
#' Callbacks can also be used with [evaluate()], in this case, the callbacks that
#' are used are equivalent to those of the validation loop when using [fit()]:
#'
#' ```
#' Start Valid
#'  - on_valid_begin
#'  Start Batch Loop
#'   - on_valid_batch_begin
#'   Start Default Validation Step
#'    - on_valid_batch_after_pred
#'    - on_valid_batch_after_loss
#'   End Default Validation Step
#'   - on_valid_batch_end
#'  End Batch Loop
#'  - on_valid_end
#' End Valid
#' ```
#'
#' @examples
#' print_callback <- luz_callback(
#'  name = "print_callback",
#'  on_train_batch_end = function() {
#'    cat("Iteration ", ctx$iter, "\n")
#'  },
#'  on_epoch_end = function() {
#'    cat("Done!\n")
#'  }
#' )
#' @returns
#' A `luz_callback` that can be passed to [fit.luz_module_generator()].
#' @family luz_callbacks
#'
#' @importFrom progress progress_bar
#' @export
luz_callback <- function(name = NULL, ..., private = NULL, active = NULL, parent_env = parent.frame(),
                         inherit = NULL) {
  make_class(
    name = name,
    ...,
    private = private,
    active = active,
    parent_env = parent_env,
    inherit = attr(inherit, "r6_class") %||% LuzCallback,
    .init_fun = TRUE
  )
}

#' Progress callback
#'
#' Responsible for printing progress during training.
#'
#' @note In general you don't need to use these callback by yourself because it's always
#'   included by default in [fit.luz_module_generator()].
#'
#' @note Printing can be disabled by passing `verbose=FALSE` to [fit.luz_module_generator()].
#'
#' @family luz_callbacks
#'
#' @returns
#' A `luz_callback`
#'
#' @export
luz_callback_progress <- luz_callback(
  "progress_callback",
  on_epoch_begin = function() {
    inform(sprintf(
      "Epoch %d/%d",
      as.integer(ctx$epoch),
      as.integer(ctx$max_epochs)
    ))
  },
  on_train_begin = function() {
    self$initialize_progress_bar("train")
  },
  on_train_batch_end = function() {
    self$tick_progress_bar("train")
  },
  on_valid_begin = function() {
    self$initialize_progress_bar("valid")
  },
  on_valid_batch_end = function() {
    self$tick_progress_bar("valid")
  },
  on_predict_begin = function() {
    self$initialize_progress_bar("predict")
  },
  on_predict_batch_end = function() {
    self$tick_progress_bar("predict")
  },
  on_epoch_end = function() {
    self$inform_metrics("train", "Train")
    self$inform_metrics("valid", "Valid")
  },
  get_abbrevs = function(metrics) {
    sapply(metrics, function(x) x$abbrev %||% class(x))
  },
  get_metrics = function(split) {

    metrics <- ctx$metrics[[split]]

    if (length(metrics) == 0)
      return(list())

    # grab pre-computed values (they might not be available though)
    values <- ctx$get_metrics(set = split, epoch = ctx$epoch)

    if (is.null(values)) {
      values <- lapply(metrics, function(x) {
        x$compute()
      })
    }

    # format
    l <- lapply(seq_along(metrics), function(i) {
      metrics[[i]]$format(values[[i]])
    })

    names(l) <- self$get_abbrevs(metrics)
    l
  },
  inform_metrics = function(split, name) {
    metrics <- self$get_metrics(split)
    if (length(metrics) > 0) {
      res <- paste0(glue::glue("{names(metrics)}: {metrics}"), collapse = " - ")
      inform(glue::glue("{name} metrics: {res}"))
    }
  },
  initialize_progress_bar = function(split) {
    total <- length(ctx$data) # ctx$data is the current dataset - can be the validation or training.

    if (!is.na(total)) {
      format <- ":current/:total [:bar]"
    } else {
      format <- ":current/unk [:spin]"
    }

    # Specially for testing purposes we don't want to have the
    # progress bar showing the ETA.
    if (getOption("luz.show_progress_bar_eta", TRUE)) {
      if (!is.na(total)) {
        format <- paste0(format, " - ETA: :eta")
      } else {
        format <- paste0(format, " - Rate: :tick_rate iter/s")
      }
    }

    metrics <- ctx$metrics[[split]]
    if (length(metrics) > 0) {
      abbrevs <- self$get_abbrevs(metrics)
      abbrevs <- paste0(glue::glue("{abbrevs}: :{tolower(abbrevs)} "), collapse = " - ")
    } else {
      abbrevs <- NULL
    }

    show_after <- if (getOption("luz.force_progress_bar", FALSE)) 0 else 0.2

    format <- paste0(c(format, abbrevs), collapse = " - ")

    self$pb <- progress::progress_bar$new(
      force = getOption("luz.force_progress_bar", FALSE),
      show_after = show_after,
      format = format,
      total = total
    )
  },
  tick_progress_bar = function(split) {
    if (ctx$verbose) {
      tokens <- self$get_metrics(split)
      names(tokens) <- tolower(names(tokens))
      self$pb$tick(tokens = tokens)
    }
  }
)

#' Metrics callback
#'
#' Tracks metrics passed to [setup()] during training and validation.
#'
#' @details This callback takes care of 2 [ctx] attributes:
#' - `ctx$metrics`: stores the current metrics objects that are initialized once for epoch,
#'   and are further `update()`d and `compute()`d every batch. You will rarely need
#'   to work with these metrics.
#' - `ctx$records$metrics`: Stores metrics per training/validation and epoch. The
#'   structure is very similar to `ctx$losses`.
#'
#' @note In general you won't need to explicitly use the metrics callback as it's
#' used by default in [fit.luz_module_generator()].
#'
#' @family luz_callbacks
#'
#' @returns
#' A `luz_callback`
#'
#' @export
luz_callback_metrics <- luz_callback(
  "metrics_callback",
  on_fit_begin = function() {
   ctx$metrics <- list(
     train = NULL,
     valid = NULL
   )
  },
  on_train_begin = function() {
    ctx$metrics$train <- lapply(
      ctx$model$metrics$train %||% list(),
      self$initialize_metric
    )
  },
  on_train_batch_end = function() {
    lapply(ctx$metrics$train, self$call_update_on_metric)
  },
  on_train_end = function() {
    self$log_all_metrics("train")
  },
  on_valid_begin = function() {
    ctx$metrics$valid <- lapply(
      ctx$model$metrics$valid %||% list(),
      self$initialize_metric
    )
  },
  on_valid_batch_end = function() {
    lapply(ctx$metrics$valid, self$call_update_on_metric)
  },
  on_valid_end = function() {
    self$log_all_metrics("valid")
  },
  initialize_metric  = function(x) {
    obj <- x$new()$to(device = ctx$device)
    bind_context(obj, ctx)
    obj
  },
  log_all_metrics = function(set) {
    lapply(
      ctx$metrics[[set]],
      function(x) {
        ctx$log_metric(tolower(x$abbrev), self$call_compute_on_metric(x))
      }
    )
  },
  call_update_on_metric = function(metric) {
    rlang::try_fetch({
      metric$update(ctx$pred, ctx$target)
    },
    error = function(cnd) {
      cli::cli_abort(
        c(
          "Error when evaluating {.field update} for metric with abbrev {.val {metric$abbrev}} and class {.cls {class(metric)}}",
          i = "The error happened at iter {.val {ctx$iter}} of epoch {.val {ctx$epoch}}.",
          i = "The model was {.emph {ifelse(ctx$training, '', 'not ')}}in training mode."
        ),
        parent = cnd
      )
    })
  },
  call_compute_on_metric = function(metric) {
    rlang::try_fetch({
      metric$compute()
    },
    error = function(cnd) {
      cli::cli_abort(
        c(
          "Error when evaluating {.field compute} for metric with abbrev {.val {metric$abbrev}} and class {.cls {class(metric)}}",
          i = "The error happened at iter {.val {ctx$iter}} of epoch {.val {ctx$epoch}}.",
          i = "The model was {.emph {ifelse(ctx$training, '', 'not ')}}in training mode."
        ),
        parent = cnd
      )
    })
  }
)

#' Train-eval callback
#'
#' Switches important flags for training and evaluation modes.
#'
#' @details It takes care of the three `ctx` attributes:
#' - `ctx$model`: Responsible for calling `ctx$model$train()` and `ctx$model$eval()`,
#'   when appropriate.
#' - `ctx$training`: Sets this flag to `TRUE` when training and `FALSE` when in
#'   validation mode.
#' - `ctx$loss`: Resets the `loss` attribute to `list()` when finished training/ or
#'   validating.
#'
#' @note In general you won't need to explicitly use the metrics callback as it's
#' used by default in [fit.luz_module_generator()].
#'
#' @returns
#' A `luz_callback`
#'
#' @family luz_callbacks
#' @export
luz_callback_train_valid <- luz_callback(
  "train_valid_callback",
  on_train_begin = function() {
    ctx$model$train()
    ctx$training <- TRUE
    ctx$loss <- list()
  },
  on_valid_begin = function() {
    ctx$model$eval()
    ctx$training <- FALSE
    ctx$loss <- list()
  }
)

#' Learning rate scheduler callback
#'
#' Initializes and runs [torch::lr_scheduler()]s.
#'
#' @param lr_scheduler A [torch::lr_scheduler()] that will be initialized with
#' the optimizer and the `...` parameters.
#' @param ... Additional arguments passed to `lr_scheduler` together with
#' the optimizers.
#' @param call_on The callback breakpoint that `scheduler$step()` is called.
#' Default is `'on_epoch_end'`. See [luz_callback()] for more information.
#' @param opt_name name of the optimizer that will be affected by this callback.
#' Should match the name given in `set_optimizers`. If your module has a single
#' optimizer, `opt_name` is not used.
#'
#' @examples
#' if (torch::torch_is_installed()) {
#' cb <- luz_callback_lr_scheduler(torch::lr_step, step_size = 30)
#' }
#' @returns
#' A [luz_callback()] generator.
#'
#' @family luz_callbacks
#' @export
luz_callback_lr_scheduler <- luz_callback(
  name = "lr_scheduler_callback",
  initialize = function(lr_scheduler, ..., call_on = "on_epoch_end",
                        opt_name = NULL) {
    self$lr_scheduler_fn <- function(optimizer) {
      lr_scheduler(optimizer, ...)
    }
    self[[call_on]] <- function() {
      if ("metrics" %in% names(formals(self$scheduler$step))) {
        current_loss <- ctx$loss[[self$opt_name]]
        self$scheduler$step(current_loss)
      } else {
        self$scheduler$step()
      }
    }
    self$opt_name <- opt_name
  },
  on_fit_begin = function() {

    if (is.null(self$opt_name) && (length(ctx$optimizers) == 1))
      self$opt_name <- names(ctx$optimizers)
    else
      rlang::abort("An optimizer name was not supplied and your model has multiple optimizers")

    if (!self$opt_name %in% names(ctx$optimizers))
      rlang::abort(glue::glue("opt_name '{self$opt_name}' not found in ctx$optimizers."))

    self$scheduler <- self$lr_scheduler_fn(ctx$optimizers[[self$opt_name]])
  },
  state_dict = function() {
    self$scheduler$state_dict()
  },
  load_state_dict = function(state_dict) {
    self$scheduler$load_state_dict(state_dict)
  }
)

#' CSV logger callback
#'
#' Logs metrics obtained during training a fiel on disk.
#' The file will have 1 line for each epoch/validation.
#'
#' @param path path to a file on disk.
#'
#' @family luz_callbacks
#' @export
luz_callback_csv_logger <- luz_callback(
  name = "csv_logger_callback",
  initialize = function(path) {
    self$path <- path.expand(path)
    self$append <- FALSE
  },
  on_epoch_end = function() {

    metrics <- rbind(
      self$to_metric_df(ctx$get_metrics("train", ctx$epoch), "train"),
      self$to_metric_df(ctx$get_metrics("valid", ctx$epoch), "valid")
    )

    utils::write.table(
      metrics,
      file = self$path,
      append = self$append,
      col.names = !self$append,
      row.names = FALSE,
      sep = ","
    )

    # now that we wrote for the first time it's ok to set append to TRUE
    self$append <- TRUE
  },
  to_metric_df = function(metrics, set) {

    if (is.null(metrics))
      return(NULL)

    metrics <- as.data.frame(metrics)
    nms <- names(metrics)
    metrics$epoch <- ctx$epoch
    metrics$set <- set
    metrics <- metrics[, c("epoch", "set", nms)]
    metrics
  }
)

#' Gradient clipping callback
#'
#' By adding the GradientClip callback, the gradient `norm_type` (default:2) norm
#' is clipped to at most `max_norm` (default:1) using [torch::nn_utils_clip_grad_norm_()],
#' which can avoid loss divergence.
#'
#' @references
#' See FastAI [documentation](https://docs.fast.ai/callback.training.html#GradientClip)
#' for the GradientClip callback.
#'
#' @inheritParams torch::nn_utils_clip_grad_norm_
#' @export
luz_callback_gradient_clip <- luz_callback(
  initialize = function(max_norm = 1, norm_type = 2) {

    if (!rlang::is_scalar_double(max_norm))
      cli::cli_abort(c(
        "{.var max_norm} should be a numeric scalar value.",
        "x" = "Got {.cls {class(max_norm)}} with length {length(max_norm)}"
      ))

    if (!rlang::is_scalar_double(norm_type))
      cli::cli_abort(c(
        "{.var norm_type} should be a numeric scalar value.",
        "x" = "Got {.cls {class(norm_type)}} with length {length(norm_type)}"
      ))

    self$max_norm <- max_norm
    self$norm_type <- norm_type
  },
  on_train_batch_before_step = function() {
    torch::nn_utils_clip_grad_norm_(
      parameters = ctx$model$parameters,
      max_norm = self$max_norm,
      norm_type = self$norm_type
    )
  }
)


