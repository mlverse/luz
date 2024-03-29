#' @include callbacks.R
NULL

#' Automatic Mixed Precision callback
#'
#' This callback will enable [torch::local_autocast()] training model forward
#' and during loss computation. It will then disable autocast and scale the loss
#' before `backward()` and `opt$step()`. See [here](https://torch.mlverse.org/docs/articles/amp.html)
#' for more information.
#'
#' @param ... Passed to [torch::cuda_amp_grad_scaler()].
#'
#' @returns
#' A `luz_callback`
#'
#' @family luz_callbacks
#'
#' @export
luz_callback_mixed_precision <- luz_callback(
  "mixed_precision_callback",
  initialize = function(...) {
    self$autocast_context <- NULL
    self$scaler <- torch::cuda_amp_grad_scaler(...)
  },
  on_fit_begin = function() {
    ctx$step_opt <- function(opt) {
      self$scaler$step(opt)
      self$scaler$update()
    }
  },
  on_train_batch_begin = function() {
    device_type <- if (grepl("cuda", ctx$device)) "cuda" else ctx$device
    self$autocast_context <- torch::set_autocast(device_type = device_type)
  },
  on_train_batch_after_loss = function() {
    torch::unset_autocast(self$autocast_context)
    self$autocast_context <- NULL
  },
  on_train_batch_before_backward = function() {
    torch::with_enable_grad({
      ctx$loss_grad <- self$scaler$scale(ctx$loss_grad)
    })
  }
)
