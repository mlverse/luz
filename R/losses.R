#' Loss to be used with `callbacks_mixup()`.
#'
#' In the training phase, computes individual losses with regard to two targets, weights them item-wise,
#' and averages the linear combinations to yield the mean batch loss.
#' For validation and testing, defers to the passed-in loss.
#'
#' It should be used together with [luz_callback_mixup()].
#'
#' @param loss the underlying loss `nn_module` to call. It must
#'   support the `reduction` field. During training the attribute will be changed to
#'   `'none'` so we get the loss for individual observations. See for for example
#'   documentation for the `reduction` argument in [torch::nn_cross_entropy_loss()].
#'
#' @seealso [luz_callback_mixup()]
#'
#' @export
nn_mixup_loss <- torch::nn_module(

  initialize = function(loss) {

    if (!inherits(loss, "nn_module"))
      rlang::abort("Loss must be a `nn_module`.")

    if (is.null(loss$reduction))
      rlang::abort("The `reduction` attribute must be supported by the `nn_module`.")

    self$loss <- loss
  },

  forward = function(input, target) {

    if (is.list(target)) {

      old_reduction <- self$loss$reduction
      self$loss$reduction <- "none"
      # Prefer the `on.exit` because the code belowe might fail and we want
      # to revert the change we made.
      on.exit({
        self$loss$reduction <- old_reduction
      }, add = TRUE)

      targets <- target[[1]]
      weight <- target[[2]]

      l1 <- self$loss(input, targets[[1]])
      l2 <- self$loss(input, targets[[2]])
      loss <- torch::torch_lerp(l1, l2, weight)

      torch::torch_mean(loss)

    } else {

      self$loss(input, target)

    }
  }
)


