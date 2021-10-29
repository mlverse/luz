#' Loss to be used with `callbacks_mixup()`.
#'
#' In the training phase, computes individual losses with regard to two targets, weights them item-wise,
#' and averages the linear combinations to yield the mean batch loss.
#' For validation and testing, defers to the passed-in loss.
#'
#' @param loss the underlying loss module to call
#' @export
nn_mixup_loss <- torch::nn_module(

  initialize = function(loss) {
    self$loss <- loss
  },

  forward = function(input, target) {

    if (is.list(target)) {

      self$loss$reduction <- "none"

      targets <- target[[1]]
      weight <- target[[2]]

      l1 <- self$loss(input, targets[[1]])
      l2 <- self$loss(input, targets[[2]])
      loss <- torch::torch_lerp(l1, l2, weight)

      self$loss$reduction <- "mean"

      torch::torch_mean(loss)

    } else {

      self$loss(input, target)

    }
  }
)


