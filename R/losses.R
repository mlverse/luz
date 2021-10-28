#' Loss to be used with `callbacks_mixup()`.
#'
#' In training phase, computes individual cross entropy losses with regard to two targets, weights them item-wise,
#' and averages the linear combinations to yield the mean batch loss.
#' For validation end testing, defers to the usual cross entropy loss.
#'
#' @inheritParams torch::nnf_cross_entropy
#' @export
luz_mixup_cross_entropy <- function(input, target, weight=NULL, ignore_index=-100, reduction=c("mean", "sum", "none")) {

  if (is.list(target) && dim(target[[1]])[2] == 2) {
    l1 <- torch::nnf_cross_entropy(input, target[[1]][ , 1], weight, ignore_index, reduction = "none")
    l2 <- torch::nnf_cross_entropy(input, target[[1]][ , 2], weight, ignore_index, reduction = "none")
    weight <- target[[2]]

    loss <- torch::torch_lerp(l1, l2, weight)
    torch::torch_mean(loss)

  } else {
    torch::nnf_cross_entropy(input, target, weight, ignore_index, reduction)
  }


}
