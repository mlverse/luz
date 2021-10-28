#' Mixup callback
#'
#' Implementation of https://arxiv.org/abs/1710.09412, restricted to categorical data.
#' The targets are supposed to be integers, not one-hot encoded vectors.
#' This callback is supposed to be used with `luz_mixup_cross_entropy` loss.
#'
#' @details
#' Overall, we follow the fastai implementation
#' (https://github.com/fastai/fastai/blob/master/fastai/callback/mixup.py)
#' described here: https://forums.fast.ai/t/mixup-data-augmentation/22764).
#' Namely,
#' - We work with a single dataloader only, randomly mixing two observations from the same batch.
#' - We linearly combine losses computed for both targets:
#'   `loss(output, new_target) = weight * loss(output, target1) + (1-weight) * loss(output, target2)`
#' - We draw different mixing coefficients for every pair.
#' - We replace `weight` with `weight = max(weight, 1-weight)` to avoid duplicates.
#'
#' @examples
#' mixup_callback <- luz_callback_mixup()
#'
#' @returns
#' A `luz_callback`
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
    xdim <- length(ctx$batch$x$size())
    xrep <- rep(1, xdim -1)
    device <- ctx$batch$y$device

    # draw mixing weights from a beta distribution with identical parameters
    weight <- rbeta(batch_len, self$alpha, self$alpha) %>% torch::torch_tensor(device = device)
    weight <- torch::torch_stack(list(weight, 1 - weight), 2)
    weight <- weight$max(2)[1][[1]]

    # determine which observations to mix
    shuffle <- torch::torch_randperm(batch_len, dtype = torch::torch_long(), device = device) + 1L

    # linearly combine the inputs according to the mixing weights ...
    x1 <- ctx$batch$x
    x2 <- ctx$batch$x[shuffle, ]
    # ... and replace the current batch input by this
    ctx$batch$x <- torch::torch_lerp(x1, x2, weight$view(list(batch_len, xrep) %>% unlist()))

    # replace current batch target with a list of:
    # 1) both targets stacked into a single tensor and
    # 2) a tensor holding the mixing weights
    y1 <- ctx$batch$y
    y2 <- self$ctx$batch$y[shuffle]
    ctx$batch$y <- list(torch::torch_stack(list(y1, y2), 2), weight)
  }
)



