#' Mixup callback
#'
#' Implementation of ['mixup: Beyond Empirical Risk Minimization'](https://arxiv.org/abs/1710.09412).
#' As of today, tested only for categorical data,
#' where targets are expected to be integers, not one-hot encoded vectors.
#' This callback is supposed to be used together with [nn_mixup_loss()].
#'
#' @details
#' Overall, we follow the [fastai implementation](https://github.com/fastai/fastai/blob/master/fastai/callback/mixup.py)
#' described [here](https://forums.fast.ai/t/mixup-data-augmentation/22764).
#' Namely,
#' - We work with a single dataloader only, randomly mixing two observations from the same batch.
#' - We linearly combine losses computed for both targets:
#'   `loss(output, new_target) = weight * loss(output, target1) + (1-weight) * loss(output, target2)`
#' - We draw different mixing coefficients for every pair.
#' - We replace `weight` with `weight = max(weight, 1-weight)` to avoid duplicates.
#'
#' @param alpha parameter for the beta distribution used to sample mixing coefficients
#'
#' @examples
#' if (torch::torch_is_installed()) {
#' mixup_callback <- luz_callback_mixup()
#' }
#'
#' @returns
#' A `luz_callback`
#'
#' @seealso [nn_mixup_loss()], [nnf_mixup()]
#'
#' @family luz_callbacks
#' @export
luz_callback_mixup <- luz_callback(
  "mixup_callback",
  initialize = function(alpha = 0.4) {
    self$alpha <- alpha
  },

  on_train_batch_begin = function() {

    batch_len <- ctx$batch$y$size(1)
    device <- ctx$batch$y$device

    # draw mixing weights from a beta distribution with identical parameters
    weight <- rbeta(batch_len, self$alpha, self$alpha) %>% torch::torch_tensor(device = device)
    weight <- torch::torch_stack(list(weight, 1 - weight), 2)
    weight <- weight$max(2)[1][[1]]

    # (1) linearly combine the inputs according to the mixing weights,
    # (2) as new target, create a list of:
    #     - (a) both targets stacked into a single tensor and
    #     - (b) a tensor holding the mixing weights,
    # (3) and replace the current batch with this
    c(mixed_x, stacked_y_with_weights) %<-% nnf_mixup(
      ctx$batch$x,
      ctx$batch$y,
      weight)

    ctx$batch$x <- mixed_x
    ctx$batch$y <- stacked_y_with_weights
  }
)

#' Mixup logic
#'
#' Logic underlying [luz_callback_mixup()].
#'
#' @details
#' Based on the passed-in input and target batches, as well as applicable mixing weights,
#' we return new tensors intended to replace the current batch.
#' The new input batch is a weighted linear combination of input batch items, while
#' the new target batch bundles the original targets, as well as the mixing weights, in
#' a nested list.
#'
#' @param x an input batch
#' @param y a target batch
#' @param weight weighting coefficient to be used by `torch_lerp()`
#'
#' @examples
#' if (torch::torch_is_installed()) {
#' batch_x <- torch::torch_randn(c(10, 768))
#' batch_y <- torch::torch_randn(10)
#' weight <- torch::torch_tensor(rep(0.9, 10))$view(c(10, 1))
#' nnf_mixup(batch_x, batch_y, weight)
#' }
#'
#' @returns
#' A `list` of:
#' - `x`, the new, mixed-up input batch
#' - `y`, a `list` of:
#'   - `ys`, a `list` of:
#'     - `y1`, the original target `y1`
#'     - `y2`, the mixed-in target `y2`
#'   - `weight`, the mixing weights
#'
#' @seealso [luz_callback_mixup()]
#'
#' @export
nnf_mixup <- function(x, y, weight) {

  # determine which observations to mix
  batch_len <- y$size(1)
  device <- y$device
  shuffle <- torch::torch_randperm(batch_len, dtype = torch::torch_long(), device = device) + 1L

  # expand weight as needed
  xdim <- length(x$size())
  xrep <- rep(1, xdim -1)
  weight <- weight$view(list(batch_len, xrep) %>% unlist())

  # create new x
  x1 <- x
  x2 <- x[shuffle, ]
  mixed_x <- torch::torch_lerp(x1, x2, weight)

  # create new y
  y1 <- y
  y2 <- y[shuffle]
  stacked_y_with_weights <- list(ys = list(y1 = y1, y2 = y2), weight = weight)

  list(x = mixed_x, y = stacked_y_with_weights)

}



