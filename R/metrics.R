#' @include utils.R
NULL

LuzMetric <- R6::R6Class(
  "LuzMetric",
  lock_objects = FALSE,
  public = list(
    format = function(v) {
      if (is.numeric(v))
        round(v, 4)
      else if (is.list(v)) {
        v <- lapply(v, round, 4)
        paste0(glue::glue("{names(v)}: {v}"), collapse = " | ")
      }
    },
    to = function(device) {
      # move tensors to the correct device
      for (nm in names(self)) {
        if (inherits(self[[nm]], "torch_tensor")) {
          if (device == "mps" && self[[nm]]$dtype == torch::torch_float64())
            self[[nm]] <- self[[nm]]$to(dtype = torch::torch_float32())

          self[[nm]] <- self[[nm]]$to(device = device)
        }
      }
      invisible(self)
    }
  )
)

#' Creates a metric set
#'
#' A metric set can be used to specify metrics that are only evaluated during
#' training, validation or both.
#'
#' @param metrics A list of luz_metrics that are meant to be used in both training
#'   and validation.
#' @param train_metrics A list of luz_metrics that are only used during training.
#' @param valid_metrics A list of luz_metrics that are only sued for validation.
#'
#' @export
luz_metric_set <- function(metrics = NULL, train_metrics = NULL, valid_metrics = NULL) {
  if (!is.null(metrics) && !(is.list(metrics) && !inherits(metrics, "luz_metric_generator")))
    metrics <- list(metrics)

  metrics <- append(list(luz_metric_loss_average()), metrics)
  new_luz_metric_set(metrics, train_metrics, valid_metrics)
}

maybe_list_metric <- function(x) {
  if (inherits(x, "luz_metric_generator"))
    list(x)
  else
    x
}

new_luz_metric_set <- function(metrics, train_metrics, valid_metrics) {
  metrics <- maybe_list_metric(metrics)
  train_metrics <- maybe_list_metric(train_metrics)
  valid_metrics <- maybe_list_metric(valid_metrics)

  sapply(metrics, assert_is_metric)
  sapply(train_metrics, assert_is_metric)
  sapply(valid_metrics, assert_is_metric)
  structure(list(
    train = c(metrics, train_metrics),
    valid = c(metrics, valid_metrics)
  ), class = "luz_metric_set")
}

assert_is_metric <- function(x) {
  if(!inherits(x, "luz_metric_generator")) {
    cli::cli_abort(c(
      "Expected an object with class {.cls luz_metric_generator}.",
      i = "Got an object with class {.cls {class(x)}}."
    ))
  }
  invisible(TRUE)
}

is_luz_metric_set <- function(obj) {
  inherits(obj, "luz_metric_set")
}

#' Creates a new luz metric
#'
#' @param name string naming the new metric.
#' @param ... named list of public methods. You should implement at least
#'  `initialize`, `update` and `compute`. See the details section for more
#'  information.
#' @inheritParams R6::R6Class
#'
#' @includeRmd man/rmd/metrics.Rmd details
#' @returns
#' Returns new luz metric.
#'
#' @examples
#' luz_metric_accuracy <- luz_metric(
#'   # An abbreviation to be shown in progress bars, or
#'   # when printing progress
#'   abbrev = "Acc",
#'   # Initial setup for the metric. Metrics are initialized
#'   # every epoch, for both training and validation
#'   initialize = function() {
#'     self$correct <- 0
#'     self$total <- 0
#'   },
#'   # Run at every training or validation step and updates
#'   # the internal state. The update function takes `preds`
#'   # and `target` as parameters.
#'   update = function(preds, target) {
#'     pred <- torch::torch_argmax(preds, dim = 2)
#'     self$correct <- self$correct + (pred == target)$
#'       to(dtype = torch::torch_float())$
#'       sum()$
#'       item()
#'     self$total <- self$total + pred$numel()
#'   },
#'   # Use the internal state to query the metric value
#'   compute = function() {
#'     self$correct/self$total
#'   }
#' )
#'
#' @export
#' @family luz_metrics
luz_metric <- function(name = NULL, ..., private = NULL, active = NULL,
                       parent_env = parent.frame(), inherit = NULL) {

  out_class <- c("luz_metric_generator", "R6ClassGenerator")
  if (!is.null(name)){
    out_class <- c(paste0(name, "_generator"), out_class)
  }

  make_class(
    name = name,
    ...,
    private = private,
    active = active,
    parent_env = parent_env,
    inherit = attr(inherit, "r6_class") %||% LuzMetric,
    .init_fun = FALSE,
    .out_class = out_class
  )
}

#' Accuracy
#'
#' Computes accuracy for multi-class classification problems.
#'
#' This metric expects to take logits or probabilities at every
#' update. It will then take the columnwise argmax and compare
#' to the target.
#'
#' @examples
#' if (torch::torch_is_installed()) {
#' library(torch)
#' metric <- luz_metric_accuracy()
#' metric <- metric$new()
#' metric$update(torch_randn(100, 10), torch::torch_randint(1, 10, size = 100))
#' metric$compute()
#' }
#' @export
#'
#'
#' @returns
#' Returns new luz metric.
#'
#' @family luz_metrics
luz_metric_accuracy <- luz_metric(
  abbrev = "Acc",
  initialize = function() {
    self$correct <- 0
    self$total <- 0
  },
  update = function(preds, target) {
    pred <- torch::torch_argmax(preds, dim = 2)
    self$correct <- self$correct + (pred == target)$
      to(dtype = torch::torch_float())$
      sum()$
      item()
    self$total <- self$total + pred$numel()
  },
  compute = function() {
    self$correct/self$total
  }
)

#' Binary accuracy
#'
#' Computes the accuracy for binary classification problems where the
#' model returns probabilities. Commonly used when the loss is [torch::nn_bce_loss()].
#'
#' @inheritParams luz_metric_binary_accuracy_with_logits
#'
#' @examples
#' if (torch::torch_is_installed()) {
#' library(torch)
#' metric <- luz_metric_binary_accuracy(threshold = 0.5)
#' metric <- metric$new()
#' metric$update(torch_rand(100), torch::torch_randint(0, 1, size = 100))
#' metric$compute()
#' }
#'
#' @returns
#' Returns new luz metric.
#'
#' @family luz_metrics
#' @export
luz_metric_binary_accuracy <- luz_metric(
  abbrev = "Acc",
  initialize = function(threshold = 0.5) {
    self$correct <- 0
    self$total <- 0
    self$threshold <- threshold
  },
  update = function(preds, targets) {
    preds <- (preds > self$threshold)$
      to(dtype = torch::torch_float())
    self$correct <- self$correct + (preds == targets)$
      to(dtype = torch::torch_float())$
      sum()$
      item()
    self$total <- self$total + preds$numel()
  },
  compute = function() {
    self$correct/self$total
  }
)

#' Binary accuracy with logits
#'
#' Computes accuracy for binary classification problems where the model
#' return logits. Commonly used together with [torch::nn_bce_with_logits_loss()].
#'
#' Probabilities are generated using [torch::nnf_sigmoid()] and `threshold` is used to
#' classify between 0 or 1.
#'
#' @param threshold value used to classifiy observations between 0 and 1.
#'
#' @examples
#' if (torch::torch_is_installed()) {
#' library(torch)
#' metric <- luz_metric_binary_accuracy_with_logits(threshold = 0.5)
#' metric <- metric$new()
#' metric$update(torch_randn(100), torch::torch_randint(0, 1, size = 100))
#' metric$compute()
#' }
#' @returns
#' Returns new luz metric.
#'
#' @family luz_metrics
#' @export
luz_metric_binary_accuracy_with_logits <- luz_metric(
  abbrev = "Acc",
  inherit = luz_metric_binary_accuracy,
  update = function(preds, targets) {
    super$update(torch::torch_sigmoid(preds), targets)
  }
)

#' Internal metric that is used to track the loss
#' @noRd
luz_metric_loss_average <- luz_metric(
  abbrev = "Loss",
  initialize = function() {
    self$values <- list()
  },
  update = function(preds, targets) {
    if (length(ctx$loss) == 1 && is.list(ctx$loss))
      loss <- ctx$loss[[1]]
    else
      loss <- ctx$loss

    self$values[[length(self$values) + 1]] <- loss
  },
  average_metric = function(x) {
    if (is.numeric(x[[1]]) || inherits(x[[1]], "torch_tensor"))
      x <- sapply(x, self$to_numeric)

    if (is.numeric(x)) {
      mean(x)
    } else if (is.list(x)) {
      lapply(purrr::transpose(x), self$average_metric)
    } else if (is.null(x)) {
      NULL
    } else {
      rlang::abort(c(
        "Average metric requires numeric tensor or values or list of them.")
      )
    }
  },
  compute = function() {
    self$average_metric(self$values)
  },
  to_numeric = function(x) {
    if (is.numeric(x))
      x
    else if (inherits(x, "torch_tensor"))
      as.numeric(x$to(device = "cpu"))
    else
      rlang::abort("Expected a numeric value or a tensor.")
  }
)

#' Mean absolute error
#'
#' Computes the mean absolute error.
#'
#'
#' @examples
#' if (torch::torch_is_installed()) {
#' library(torch)
#' metric <- luz_metric_mae()
#' metric <- metric$new()
#' metric$update(torch_randn(100), torch_randn(100))
#' metric$compute()
#' }
#' @returns
#' Returns new luz metric.
#'
#' @family luz_metrics
#' @export
luz_metric_mae <- luz_metric(
  abbrev = "MAE",
  initialize = function() {
    self$sum_abs_error <- torch::torch_tensor(0, dtype = torch::torch_float64())
    self$n <- torch::torch_tensor(0, dtype = torch::torch_int64())
  },
  update = function(preds, targets) {
    self$sum_abs_error <- self$sum_abs_error + torch::torch_sum(torch::torch_abs(preds - targets))$
      to(device = "cpu")
    self$n <- self$n + targets$numel()
  },
  compute = function() {
    (self$sum_abs_error / self$n)$item()
  }
)

#' Mean squared error
#'
#' Computes the mean squared error
#'
#' @returns
#' A luz_metric object.
#'
#' @family luz_metrics
#' @export
luz_metric_mse <- luz_metric(
  abbrev = "MSE",
  initialize = function() {
    self$sum_error <- torch::torch_tensor(0, dtype = torch::torch_float64())
    self$n <- torch::torch_tensor(0, dtype = torch::torch_int64())
  },
  update = function(preds, targets) {
    self$sum_error <- self$sum_error + torch::torch_sum(torch::torch_pow(exponent = 2, preds - targets))
    self$n <- self$n + targets$numel()
  },
  compute = function() {
    (self$sum_error / self$n)$item()
  }
)

#' Root mean squared error
#'
#' Computes the root mean squared error.
#'
#' @family luz_metrics
#'
#' @returns
#' Returns new luz metric.
#'
#' @export
luz_metric_rmse <- luz_metric(
  inherit = luz_metric_mse,
  abbrev = "RMSE",
  compute = function() {
    sqrt(super$compute())
  }
)
