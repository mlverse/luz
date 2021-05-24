#' @include metrics.R
NULL

luz_metric_auc_base <- luz_metric(
  initialize = function(num_thresholds=200,
                        thresholds=NULL,
                        from_logits=FALSE) {
    if (!is.null(thresholds)) {
      self$num_thresholds <- length(thresholds) + 2
      thresholds <- sort(thresholds)
    } else {
      if (num_thresholds <= 1)
        rlang::abort("num_thresholds must be > 1")

      # Otherwise, linearly interpolate (num_thresholds - 2) thresholds in
      # (0, 1).
      self$num_thresholds <- num_thresholds
      thresholds = (2:(num_thresholds-1))/(num_thresholds-2)
    }

    # Add an endpoint "threshold" below zero and above one for either
    # threshold method to account for floating point imprecisions.
    eps <- torch::torch_finfo(torch::torch_float32())$eps
    self$thresholds = torch::torch_tensor(c(0.0 - eps, thresholds, 1.0 + eps))

    self$from_logits <- from_logits

    # Initialize state
    self$initialize_state()
  },
  initialize_state = function() {
    rlang::abort("not implemented")
  }
)

#' Computes the area under the ROC
#'
#' To avoid storing all predictions and targets for an epoch we compute confusion
#' matrices across a range of pre-established thresholds.
#'
#' @param num_thresholds Number of thresholds used to compute confusion matrices.
#'  In that case, thresholds are created by getting `num_thresholds` values linearly
#'  spaced in the unit interval.
#' @param thresholds (optional) If threshold are passed, then those are used to compute the
#'  confusion matrices and `num_thresholds` is ignored.
#' @param from_logits Boolean indicating if predictions are logits, in that case
#'  we use sigmoid to put them in the unit interval.
#'
#' @examples
#' if (torch::torch_is_installed()){
#' actual <- c(1, 1, 1, 0, 0, 0)
#' predicted <- c(0.9, 0.8, 0.4, 0.5, 0.3, 0.2)
#'
#' y_true <- torch_tensor(actual)
#' y_pred <- torch_tensor(predicted)
#'
#' m <- luz_metric_binary_auroc(thresholds = predicted)
#' m <- m$new()
#'
#' m$update(y_pred[1:2], y_true[1:2])
#' m$update(y_pred[3:4], y_true[3:4])
#' m$update(y_pred[5:6], y_true[5:6])
#'
#' m$compute()
#' }
#' @family luz_metrics
#' @export
luz_metric_binary_auroc <- luz_metric(
  abbrev = "AUC",
  inherit = luz_metric_auc_base,
  initialize_state = function() {
    self$true_positives <- torch::torch_zeros(size = self$num_thresholds)
    self$false_positives <- torch::torch_zeros(size = self$num_thresholds)
    self$n_pos <- torch::torch_zeros(size = 1)
    self$n_neg <- torch::torch_zeros(size = 1)
  },
  update = function(preds, targets) {

    if (self$from_logits)
      preds <- torch::torch_sigmoid(preds)

    if (preds$ndim > 1)
      preds <- preds$view(-1)

    if (targets$ndim > 1)
      targets <- targets$view(-1)

    predictions_per_thresh <- preds$unsqueeze(2) > self$thresholds
    comparisons <- predictions_per_thresh == targets$unsqueeze(2)

    self$true_positives$add_(torch_sum(comparisons[targets == 1,], dim = 1))
    self$false_positives$add_(torch_sum(comparisons[targets == 0,], dim = 1))
    self$n_pos$add_(torch_sum(targets))
    self$n_neg$add_(torch_sum(targets == 0))
  },
  compute = function() {
    tpr <- self$true_positives/self$n_pos
    fpr <- self$false_positives/self$n_neg
    mult <- torch:::torch_diff(fpr, n=1, prepend = 0)

    torch_sum(tpr*mult)$item()
  }
)

#' Computes the multi-class AUROC
#'
#' The same definition as [Keras](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC)
#' is used by default. This is equivalent to the `'micro'` method in SciKit Learn
#' too. See [docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
#'
#' **Note** that class imbalance can affect this metric unlike
#' the AUC for binary classification.
#'
#' @inheritParams luz_metric_binary_auroc
#' @param from_logits If `TRUE` then we call [torch::nnf_softmax()] in the predictions
#'   before computing the metric.
#' @param average The averaging method:
#'   - `'micro'`: Stack all classes and computes the AUROC as if it was a binary
#'     classification problem.
#'   - `'macro'`: Finds the AUCROC for each class and computes their mean.
#'   - `'weighted'`: Finds the AUROC for each class and computes their weighted
#'     mean pondering by the number of instances for each class.
#'   - `'none'`: Returns the AUROC for each class in a list.
#'
#' @details
#' Currently the AUC is approximated using the 'interpolation' method described in
#' [Keras](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC).
#'
#' @examples
#' if (torch::torch_is_installed()) {
#' actual <- c(1, 1, 1, 0, 0, 0) + 1L
#' predicted <- c(0.9, 0.8, 0.4, 0.5, 0.3, 0.2)
#' predicted <- cbind(1-predicted, predicted)
#'
#' y_true <- torch_tensor(as.integer(actual))
#' y_pred <- torch_tensor(predicted)
#'
#' m <- luz_metric_multiclass_auroc(thresholds = as.numeric(predicted),
#'                                  average = "micro")
#' m <- m$new()
#'
#' m$update(y_pred[1:2,], y_true[1:2])
#' m$update(y_pred[3:4,], y_true[3:4])
#' m$update(y_pred[5:6,], y_true[5:6])
#' m$compute()
#' }
#' @family luz_metrics
#' @export
luz_metric_multiclass_auroc <- luz_metric(
  abbrev = "AUC",
  inherit = luz_metric_auc_base,
  initialize = function(num_thresholds=200,
                        thresholds=NULL,
                        from_logits=FALSE,
                        average = c("micro", "macro", "weighted", "none")) {

    self$average <- rlang::arg_match(average)
    super$initialize(num_thresholds = num_thresholds,
                     thresholds = thresholds,
                     from_logits = from_logits)
  },
  initialize_state = function(num_classes) {

    if (missing(num_classes))
      return()

    self$num_classes <- num_classes
    size <- c(self$num_classes, self$num_thresholds)

    self$true_positives <- torch::torch_zeros(size = size)
    self$false_positives <- torch::torch_zeros(size = size)
    self$n_pos <- torch::torch_zeros(size = c(self$num_classes, 1))
    self$n_neg <- torch::torch_zeros(size = c(self$num_classes, 1))
  },
  update = function(preds, targets) {

    # initialize state. we expect one column for each class
    if (is.null(self$num_classes)) {
      if (self$average == "micro") {
        self$initialize_state(1L)
      } else {
        self$initialize_state(preds$shape[2])
      }
    }

    if (self$from_logits)
      preds <- torch::nnf_softmax(preds)

    # targets are expected to be integers representing the classes
    # so we one hot encode them
    targets <- nnf_one_hot(targets, num_classes = preds$shape[2])

    if (self$average == "micro") {
      preds <- micro_stack(preds)
      targets <- micro_stack(targets)
    }

    # usqueeze dims
    preds <- preds$unsqueeze(3)
    targets <- targets$unsqueeze(3)

    predictions_per_thresh <- preds > self$thresholds
    comparisons <- predictions_per_thresh == targets

    self$true_positives$add_(torch_sum(comparisons * (targets == 1), dim = 1))
    self$false_positives$add_(torch_sum(comparisons * (targets == 0), dim = 1))
    self$n_pos$add_(torch_sum(targets == 1, dim = 1))
    self$n_neg$add_(torch_sum(targets == 0, dim = 1))
  },
  compute = function() {

    tpr <- self$true_positives/self$n_pos
    fpr <- self$false_positives/self$n_neg

    mult <- torch::torch_diff(fpr, n=1, dim = 2,prepend = torch_zeros(self$num_classes,1))
    mult <- torch_cat(list(mult, torch_zeros(self$num_classes, 1)), dim = 2)

    aucs_minor <- torch::torch_sum(tpr*mult[,1:-2], dim = 2)
    aucs_major <- torch::torch_sum(tpr*mult[,2:N], dim = 2)

    aucs <- (aucs_minor + aucs_major)/2

    if (self$average == "none") {
      as.list(as.numeric(aucs))
    } else if (self$average == "micro") {
      aucs$item()
    } else if (self$average == "macro") {
      torch::torch_mean(aucs)$item()
    } else if (self$average == "weighted") {
      w <- self$n_pos
      w <- w/torch::torch_sum(w)
      torch::torch_sum(w$squeeze(2) * aucs)$item()
    }
  }
)

# stacking for the micro method
micro_stack <- function(x) {
  x <- torch::torch_unbind(x, dim = 2)
  x <- torch::torch_cat(x, dim = 1)
  x$unsqueeze(2)
}

