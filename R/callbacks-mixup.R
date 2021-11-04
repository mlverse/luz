#' Mixup callback
#'
#' Implementation of ['mixup: Beyond Empirical Risk Minimization'](https://arxiv.org/abs/1710.09412).
#' As of today, tested only for categorical data,
#' where targets are expected to be integers, not one-hot encoded vectors.
#' This callback is supposed to be used together with [nn_mixup_loss()].
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
#' @param alpha parameter for the beta distribution used to sample mixing coefficients
#'
#' @examples
#' mixup_callback <- luz_callback_mixup()
#'
#' @returns
#' A `luz_callback`
#'
#' @seealso [nn_mixup_loss()]
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

    # (1) linearly combine the inputs according to the mixing weights,
    # (2) as new target, create a list of:
    #     - (a) both targets stacked into a single tensor and
    #     - (b) a tensor holding the mixing weights,
    # (3) and replace the current batch with this
    c(mixed_x, stacked_y_with_weights) %<-% mixup(
      ctx$batch$x,
      ctx$batch$y,
      shuffle,
      weight$view(list(batch_len, xrep) %>% unlist()))

    ctx$batch$x <- mixed_x
    ctx$batch$y <- stacked_y_with_weights
  }
)

#' @export
mixup <- function(x, y, shuffle, weight) {

  x1 <- x
  x2 <- x[shuffle, ]
  mixed_x <- torch::torch_lerp(x1, x2, weight)

  y1 <- y
  y2 <- y[shuffle]
  stacked_y_with_weights <- list(list(y1, y2), weight)

  list(mixed_x, stacked_y_with_weights)

}



