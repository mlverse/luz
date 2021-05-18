#' @include metrics.R

#' @title Computes the area under the ROC
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
      thresholds = (1:(num_thresholds-1))/(num_thresholds-1)
    }

    # Add an endpoint "threshold" below zero and above one for either
    # threshold method to account for floating point imprecisions.
    eps <- torch::torch_finfo(torch::torch_float32())$eps
    self$thresholds = torch::torch_tensor(c(0.0 - eps, thresholds, 1.0 + eps))

    self$from_logits <- from_logits

    # Initialize state
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

