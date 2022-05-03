#' @include callbacks.R
NULL

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

    out <- ctx$get_metric(qty, set, ctx$epoch)

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
#'   call callbacks as soon as the model stops training.
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
  weight = Inf,
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

    if (is.null(self$current_best)) {
      self$current_best <- qty
      # in the first epoch we should just save the value as the current best.
      return(invisible(NULL))
    }

    if (self$compare(qty, self$current_best)) {
      # means that new qty is better then previous
      self$current_best <- qty
      self$patience_counter <- 0L
    } else {
      # mean that qty did not improve
      self$patience_counter <- self$patience_counter + 1L
    }

    if (self$patience_counter >= self$patience &&
        ctx$epoch >= ctx$min_epochs) {
      rlang::signal("Early stopping", class = "early_stopping")
    }

  },
  on_early_stopping = function() {
    inform(
      glue::glue("Early stopping at epoch {ctx$epoch} of {ctx$max_epochs}")
    )
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
#' luz_callback_model_checkpoint(path= "path/to/dir")
#' luz_callback_model_checkpoint(path= "path/to/dir/epoch-{epoch:02d}/model.pt")
#' luz_callback_model_checkpoint(path= "path/to/dir/epoch-{epoch:02d}/model-{monitor:.2f}.pt")
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
      if (self$compare(qty, self$current_best) || ctx$epoch == 1) {
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

#' Use best model
#'
#' Each epoch, if there's improvement in the monitored metric we serialize the
#' model weights to a temp file. When training is done, we reload weights from
#' the best model.
#'
#' @inheritParams luz_callback_early_stopping
#'
#' @examples
#' cb <- luz_callback_use_best_model()
#'
#' @family luz_callbacks
#' @export
luz_callback_use_best_model <- luz_callback(
  "use_best_model_callback",
  inherit = luz_callback_model_checkpoint,
  initialize = function(monitor = "valid_loss", mode="min", min_delta = 0) {
    self$path <- tempfile(fileext = "pt")
    super$initialize(
      self$path, monitor = monitor, mode = mode, min_delta = min_delta,
      save_best_only = TRUE
    )
  },
  on_fit_end = function() {
    weights <- torch::torch_load(self$path)$state_dict()
    ctx$model$load_state_dict(weights)
  }
)
