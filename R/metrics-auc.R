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

luz_metric_multiclass_auc <- luz_metric(
  abbrev = "AUC",
  inherit = luz_metric_auc_base,
  initialize_state = function(num_classes) {

    if (missing(num_classes))
      return()

    self$num_classes <- num_classes
    size <- c(self$num_classes, self$num_thresholds)

    self$true_positives <- torch::torch_zeros(size = size)
    self$false_positives <- torch::torch_zeros(size = size)
    self$n_pos <- torch::torch_zeros(size = self$num_classes)
    self$n_neg <- torch::torch_zeros(size = self$num_classes)
  },
  update = function(preds, targets) {

    # initialize state. we expect one column for each class
    if (is.null(self$num_classes))
      self$initialize_state(preds$shape[2])

    if (self$from_logits)
      preds <- torch::nnf_softmax(preds)

    # targets are expected to be integers representing the classes
    # so we one hot encode them
    targets <- nnf_one_hot(targets, num_classes = self$num_classes)

    # usqueeze dims
    preds <- preds$unsqueeze(3)
    targets <- targets$unsqueeze(3)

    predictions_per_thresh <- preds > self$thresholds
    comparisons <- predictions_per_thresh == targets

    self$true_positives$add_(torch_sum(comparisons * (targets == 1), dim = 1))
    self$false_positives$add_(torch_sum(comparisons * (targets == 0), dim = 1))
    self$n_pos$add_(torch_sum(targets == 1, dim = 1)$squeeze(2))
    self$n_neg$add_(torch_sum(targets == 0, dim = 1)$squeeze(2))
  },
  compute = function() {

    tpr <- self$true_positives/self$n_pos$unsqueeze(2)
    fpr <- self$false_positives/self$n_neg$unsqueeze(2)
    mult <- torch::torch_diff(fpr, n=1, prepend = torch_zeros(2,1), dim = 2)

    torch_mean(torch_sum(tpr*mult,dim=2))$item()
  }
)

