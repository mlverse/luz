#' Label smoothing cross entropy loss
#'
#' Label smoothing, following the implementation in the fastai library (fastai/losses.py).
#'
#' @inheritParams torch::nnf_cross_entropy
#' @export
luz_label_smoothing_cross_entropy <- function(
    input, target, weight=NULL, ignore_index=-100, reduction=c("mean", "sum", "none")) {

  reduction <- match.arg(reduction)
  c <- input$size()[2]
  eps <- 0.1

  log_preds <- torch::nnf_log_softmax(input, dim = 2)

  if (reduction == "sum") {
    loss <- -log_preds$sum()
  } else {
    loss <- -log_preds$sum(dim = 2) # We divide by that size at the return line so sum and not mean
    if (reduction == "mean") loss <- loss$mean()
  }

  loss * eps/c + (1 - eps) * torch::nnf_nll_loss(
    log_preds, target$to(dtype = torch::torch_long()), weight = weight, reduction = reduction)

}
