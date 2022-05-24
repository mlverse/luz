lr_anneal <- torch::lr_scheduler(
  "lr_lambda",
  initialize = function(
    optimizer,
    start_lr = 1e-7,
    end_lr = 1e-1,
    n_iters = 100,
    log_spaced_intervals = TRUE,
    last_epoch=-1,
    verbose=FALSE) {

    self$optimizer <- optimizer
    self$end_lr <- end_lr
    self$base_lrs <- start_lr
    self$iters <- n_iters
    self$multiplier <- (end_lr/start_lr)^(1/n_iters)
    self$log_spaced_intervals <- log_spaced_intervals

    super$initialize(optimizer, last_epoch, verbose)

  },
  get_lr = function() {
    if (self$last_epoch > 0) {
      lrs <- numeric(length(self$optimizer$param_groups))
      for (i in seq_along(self$optimizer$param_groups)) {
        if (self$log_spaced_intervals) {
          lrs[i] <- self$optimizer$param_groups[[i]]$lr * self$multiplier
        } else {
          lrs[i] <- self$optimizer$param_groups[[i]]$lr * (self$end_lr / self$optimizer$param_groups[[i]]$lr) ^ (self$last_epoch / self$iters)
        }

      }
    } else {
      lrs <- as.numeric(self$base_lrs)
    }
    lrs
  }
)


luz_callback_record_lr <- luz_callback(
  name = "luz_callback_profile_lr",
  initialize = function(steps, verbose) {
    self$total_steps <- steps
    self$steps <- 0
    self$verbose <- if (is.null(verbose)) interactive() else verbose
    if (self$verbose) {
      self$pb <- cli::cli_progress_bar(
        name = "Finding the learning rate",
        total = self$total_steps,
        type = "iterator",
        .envir = self
      )
    }
  },
  on_train_batch_end = function() {
    loss <- ctx$loss$opt$cpu()$item()
    ctx$log("lr_finder", "lr", ctx$optimizers$opt$param_groups[[1]]$lr)
    ctx$log("lr_finder", "loss", loss)
    self$steps <- self$steps + 1
    if (self$verbose) cli::cli_progress_update(id = self$pb)
    if (self$steps >= self$total_steps) {
      if (self$verbose) cli::cli_progress_done(id = self$pb)
      rlang::interrupt()
    }
  }
)


#' Learning Rate Finder
#'
#' @param object An nn_module that has been setup().
#' @param data (dataloader) A dataloader created with torch::dataloader()  used for learning rate finding.
#' @param steps (integer) The number of steps to iterate over in the learning rate finder. Default: 100.
#' @param start_lr (float) The smallest learning rate. Default: 1e-7.
#' @param end_lr (float) The highest learning rate. Default: 1e-1.
#' @param log_spaced_intervals (logical) Whether to divide the range between start_lr and end_lr into log-spaced intervals (alternative: uniform intervals). Default: TRUE
#' @param ... Other arguments passed to `fit`.
#'
#' @examples
#' if (torch::torch_is_installed()) {
#' library(torch)
#' ds <- torch::tensor_dataset(x = torch_randn(100, 10), y = torch_randn(100, 1))
#' dl <- torch::dataloader(ds, batch_size = 32)
#' model <- torch::nn_linear
#' model <- model %>% setup(
#'   loss = torch::nn_mse_loss(),
#'   optimizer = torch::optim_adam
#' ) %>%
#'   set_hparams(in_features = 10, out_features = 1)
#' records <- lr_finder(model, dl, verbose = FALSE)
#' plot(records)
#' }
#' @returns A dataframe with two columns: learning rate and loss
#' @export
lr_finder <- function(object, data, steps = 100, start_lr = 1e-7, end_lr = 1e-1,
                      log_spaced_intervals = TRUE, ..., verbose = NULL) {

  scheduler <- luz_callback_lr_scheduler(
    lr_anneal,
    verbose=FALSE,
    start_lr = start_lr,
    end_lr = end_lr,
    n_iters = steps,
    log_spaced_intervals = log_spaced_intervals,
    call_on="on_train_batch_begin"
  )

  lr_profiler <- luz_callback_record_lr(steps, verbose)

  fitted <- object %>%
    set_opt_hparams(lr = start_lr) %>%
    fit(...,
        data = data,
        epochs = 999999, # the callback will be responsible for interrupting
        callbacks = list(scheduler, lr_profiler),
        verbose = FALSE
    )

  lr_records <- data.frame(sapply(fitted$records$lr_finder, as.numeric))

  class(lr_records) <- c("lr_records", class(lr_records))
  lr_records
}

#' @export
print.lr_records <- function(x, ...) {
  NextMethod()
}

#' @export
plot.lr_records <- function(x, ...) {
  rlang::check_installed("ggplot2")

  x <- as.data.frame(x)

  loss_exp_avg <- 0
  beta <- 0.9

  for (i in 1:nrow(x)) {
    loss_exp_avg <- beta * loss_exp_avg + (1 - beta) * x$loss[i]
    x$smoothed_loss[i] <- loss_exp_avg / (1 - beta^i)
  }

  ggplot2::ggplot(x, ggplot2::aes_string(x = "lr")) +
    ggplot2::geom_line(ggplot2::aes_string(y = "loss"), linetype="dotted", size = 0.7) +
    ggplot2::geom_line(ggplot2::aes_string(y = "smoothed_loss"), color="cyan", size = 1) +
    ggplot2::scale_x_log10() +
    ggplot2::xlab("Learning Rate") +
    ggplot2::ylab("Loss")
}
