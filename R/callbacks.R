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

call_all_callbacks <- function(callbacks, name) {
  lapply(callbacks, function(callback) {
    callback$call(name)
  })
}

default_callbacks <- function() {
  list(
    luz_callback_train_valid(),
    luz_callback_metrics(),
    luz_callback_progress()
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
#' @export
luz_callback_progress <- luz_callback(
  "progress_callback",
  on_train_begin = function() {
    format <- ":current/:total [:bar] - ETA: :eta"
    metrics <- ctx$metrics[["train"]][[ctx$epoch]]
    if (length(metrics) > 0) {
      abbrevs <- self$get_abbrevs(metrics)
      abbrevs <- paste0(glue::glue("{abbrevs}: :{tolower(abbrevs)} "), collapse = " - ")
    } else {
      abbrevs <- NULL
    }

    format <- paste0(c(format, abbrevs), collapse = " - ")
    self$pb <- progress::progress_bar$new(
      format = format,
      total = length(ctx$data)
    )
  },
  on_epoch_begin = function() {
    inform(sprintf(
      "Epoch %d/%d",
      as.integer(ctx$epoch),
      as.integer(ctx$epochs)
    ))
  },
  on_train_batch_end = function() {
    if (ctx$verbose) {
      tokens <- self$get_metrics("train")
      names(tokens) <- tolower(names(tokens))
      self$pb$tick(tokens = tokens)
    }
  },
  on_epoch_end = function() {
    self$inform_metrics("train", "Train")
    self$inform_metrics("valid", "Valid")
  },
  get_abbrevs = function(metrics) {
    sapply(metrics, function(x) x$abbrev %||% class(x))
  },
  get_metrics = function(split) {

    metrics_split <- ctx$metrics[[split]]
    if (length(metrics_split) >= ctx$epoch) {
      metrics <- ctx$metrics[[split]][[ctx$epoch]]
    } else {
      return(list())
    }

    if (length(metrics) == 0)
      return(list())

    # grab pre-computed values (they might not be available though)
    metric_record <- ctx$records$metrics[[split]]
    if (length(metric_record) >= ctx$epoch) {
      values <- ctx$records$metrics[[split]][[ctx$epoch]]
    } else {
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
  }
)

#' Metrics callback
#'
#' Tracks metrics passed to [setup()] during training and validation.
#'
#' @details This callback takes care of 2 `ctx` attributes:
#' - `ctx$metrics`: stores the metrics objects that are initialized once for epoch,
#'   and are further `update()`d and `compute()`d every batch. You will rarely need
#'   to work with these metrics.
#' - `ctx$records$metrics`: Stores metrics per training/validation and epoch. The
#'   structure is very similar to `ctx$losses`.
#'
#' @note In general you won't need to explicitly use the metrics callback as it's
#' used by default in [fit.luz_module_generator()].
#'
#' @family luz_callbacks
#' @export
luz_callback_metrics <- luz_callback(
  "metrics_callback",
  on_fit_begin = function() {
   ctx$metrics <- list(
     train = list(),
     valid = list()
   )
   ctx$records$metrics <- list(
     train = list(),
     valid = list()
   )
  },
  on_train_begin = function() {
    ctx$metrics$train[[ctx$epoch]] <- lapply(
      ctx$model$metrics %||% list(),
      self$initialize_metric
    )
  },
  on_train_batch_end = function() {
    lapply(
      ctx$metrics$train[[ctx$epoch]],
      function(x) x$update(ctx$pred, ctx$target)
    )
  },
  on_train_end = function() {
    ctx$records$metrics$train[[ctx$epoch]] <- lapply(
      ctx$metrics$train[[ctx$epoch]],
      function(x) x$compute()
    )
    names(ctx$records$metrics$train[[ctx$epoch]]) <- sapply(
      ctx$metrics$train[[ctx$epoch]],
      function(x) tolower(x$abbrev)
    )
  },
  on_valid_begin = function() {
    ctx$metrics$valid[[ctx$epoch]] <- lapply(
      ctx$model$metrics %||% list(),
      self$initialize_metric
    )
  },
  on_valid_batch_end = function() {
    lapply(
      ctx$metrics$valid[[ctx$epoch]],
      function(x) x$update(ctx$pred, ctx$target)
    )
  },
  on_valid_end = function() {
    ctx$records$metrics$valid[[ctx$epoch]] <- lapply(
      ctx$metrics$valid[[ctx$epoch]],
      function(x) x$compute()
    )
    names(ctx$records$metrics$valid[[ctx$epoch]]) <- sapply(
      ctx$metrics$valid[[ctx$epoch]],
      function(x) tolower(x$abbrev)
    )
  },
  initialize_metric  = function(x) {
    obj <- x$new()
    bind_context(obj, ctx)
    obj
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

monitor_metrics <- luz_callback(
  name = "monitor_metrics",
  initialize = function(monitor, mode, min_delta) {
    self$monitor <- monitor
    self$mode <- mode
    self$min_delta <- min_delta
  },
  find_quantity = function() {

    o <- strsplit(self$monitor, "_")[[1]]
    set <- o[[1]]
    qty <- o[[2]]
    opt <- if (length(o) >= 3) o[[3]] else NULL

    out <- ctx$records$metrics[[set]][[ctx$epoch]][[qty]]

    if (!is.null(opt))
      out <- out[[opt]]

    if (length(out) != 1)
      rlang::abort(glue::glue("Expected monitored metric to be length 1, got {length(out)}"))

    out
  },
  # returns TRUE when the new is better then previous acording to mode
  compare = function(new, old) {
    out <- if (self$mode == "min")
      (old - self$min_delta) > new
    else if (self$mode == "max")
      (new - self$min_delta) > old
    else if (self$mode == "zero")
      (abs(old) - self$min_delta) > abs(self$min_delta)

    as.array(out)
  }
)

#' Early stopping callback
#'
#' Stops training when a monitored metric stops improving
#'
#' @param monitor A string in the format `<set>_<metric>` where `<set>` can be
#'  'train' or 'valid' and `<metric>` can be the abbreviation of any metric
#'  that you are tracking during training. The metric name is case insensitive.
#' @param min_delta Minimum improvement to reset the patience counter.
#' @param patience Number of epochs without improving until stoping training.
#' @param mode Specifies the direction that is considered an improvement. By default
#'  'min' is used. Can also be 'max' (higher is better) and 'zero'
#'  (closer to zero is better).
#' @param baseline An initial value that will be used as the best seen value
#'  in the begining. Model will stopm training if no better than baseline value
#'  is found in the first `patience` epochs.
#'
#' @note
#' This callback adds a `on_early_stopping` callback that can be used to
#' call callbacks after as soon as the model stopped training.
#'
#' @note
#' If `verbose=TRUE` in [fit.luz_module_generator()] a message is printed when
#' early stopping.
#'
#' @returns
#' A `luz_callback` that does early stopping.
#'
#' @examples
#' cb <- luz_callback_early_stopping()
#'
#' @family luz_callbacks
#' @export
luz_callback_early_stopping <- luz_callback(
  name = "early_stopping_callback",
  inherit = monitor_metrics,
  initialize = function(monitor = "valid_loss", min_delta = 0, patience = 0,
                        mode="min", baseline=NULL) {

    super$initialize(monitor, mode, min_delta)

    self$patience <- patience
    self$baseline <- baseline

    if (!is.null(self$baseline))
      self$current_best <- baseline

    self$patience_counter <- 0L
  },
  on_fit_begin = function() {
    ctx$handlers <- append(ctx$handlers, list(
      early_stopping = function(err) {
        ctx$call_callbacks("on_early_stopping")
        invisible(NULL)
      }
    ))
  },
  on_epoch_end = function() {

    qty <- self$find_quantity()
    if (is.null(self$current_best))
      self$current_best <- qty

    if (self$compare(qty, self$current_best)) {
      # means that new qty is better then previous
      self$current_best <- qty
      self$patience_counter <- 0L
    } else {
      # mean that qty did not improve
      self$patience_counter <- self$patience_counter + 1L
    }

    if (self$patience_counter >= self$patience) {
      rlang::signal("Early stopping", class = "early_stopping")
    }

  },
  on_early_stopping = function() {
    inform(glue::glue("Early stopping at epoch {ctx$epoch} of {ctx$epochs}"))
  }
)

#' Checkpoints model weights
#'
#' This saves checkpoints of the model according to the specified metric and
#' behavior.
#'
#' @param path Path to save the model on disk. The path is interpolated with `glue`,
#' so you can use any attribute within the [ctx] by using `'{ctx$epoch}'`. Specially
#' the `epoch` and `monitor` quantities are already in the environment. If the specified
#' path is a path to a directory (ends with `/` or `\`), then models are saved with the name given by
#' `epoch-{epoch:02d}-{self$monitor}-{monitor:.3f}.pt`. See more in the examples.
#' You can use [sprintf()] to quickly format quantities, for example:`'{epoch:02d}'`.
#' @inheritParams luz_callback_early_stopping
#' @param save_best_only if `TRUE` models are only saved if they have an improvement
#' over a previously saved model.
#' @param min_delta Minimum difference to consider as improvement. Only used when
#' `save_best_only=TRUE`.
#'
#' @note `mode` and `min_delta` are only used when `save_best_only=TRUE`.
#' `save_best_only` will overwrite the saved models if the `path` parameter
#' don't differentiate by epochs.
#'
#' @examples
#' luz_callback_checkpoint(path= "path/to/dir")
#' luz_callback_checkpoint(path= "path/to/dir/epoch-{epoch:02d}/model.pt")
#' luz_callback_checkpoint(path= "path/to/dir/epoch-{epoch:02d}/model-{monitor:.2f}.pt")
#'
#' @family luz_callbacks
#' @export
luz_callback_model_checkpoint <- luz_callback(
  name = "model_checkpoint_callback",
  inherit = monitor_metrics,
  initialize = function(path, monitor = "valid_loss", save_best_only = FALSE,
                        mode = "min", min_delta = 0) {

    if (grepl("/$", path) || grepl("\\\\$", path)) {
      path <- paste0(path, "epoch-{epoch:02d}-{self$monitor}-{monitor:.3f}.pt")
    }

    self$path <- path
    self$save_best_only <- save_best_only

    if (self$save_best_only)
      self$current_best <- 0

    super$initialize(monitor, mode, min_delta)
  },
  on_epoch_end = function() {

    qty <- self$find_quantity()
    if (is.null(self$current_best))
      self$current_best <- qty

    monitor <- qty
    epoch <- ctx$epoch

    path <- self$fmt_path(self$path)

    if (self$save_best_only) {
      if (self$compare(qty, self$current_best)) {
        # means that new qty is better then previous
        self$current_best <- qty
        fs::dir_create(fs::path_dir(path), recurse = TRUE)
        fs::file_create(path)
        luz_save_model_weights(ctx, path)
      }
    } else {
      fs::dir_create(fs::path_dir(path), recurse = TRUE)
      fs::file_create(path)
      luz_save_model_weights(ctx, path)
    }
  },
  fmt_path = function(path) {
    glue::glue(path, .transformer = sprintf_transformer, .envir = rlang::caller_env())
  }
)


