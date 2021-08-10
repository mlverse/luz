lr_anneal <- torch::lr_scheduler(
  "lr_lambda",
  initialize = function(
    optimizer,
    start_lr = 1e-7,
    end_lr = 1e-4,
    n_iters = 100,
    last_epoch=-1,
    verbose=FALSE) {

    self$optimizer <- optimizer
    self$end_lr <- end_lr
    self$pct <- pct
    self$base_lrs <- start_lr
    self$iters <- n_iters

    super$initialize(optimizer, last_epoch, verbose)

  },
  get_lr = function() {
    if (self$last_epoch > 0) {
      lrs <- numeric(length(self$optimizer$param_groups))
      for (i in seq_along(self$optimizer$param_groups)) {
        lrs[i] <- self$optimizer$param_groups[[i]]$lr * (self$end_lr / self$optimizer$param_groups[[i]]$lr) ^ (self$last_epoch / self$iters)
      }
    } else {
      lrs <- as.numeric(self$base_lrs)
    }
    lrs
  }
)


luz_callback_record_lr <- luz_callback(
  name = "luz_callback_profile_lr",
  on_train_batch_end = function() {
    loss <- as_array(ctx$loss$opt$cpu())
    ctx$log("lr_finder", "lr", ctx$optimizers$opt$param_groups[[1]]$lr)
    ctx$log("lr_finder", "loss", loss)
  }
)


#' Learning Rate Finder
#'
#' @param object An nn_module that has been setup().
#' @param data (dataloader) A dataloader created with torch::dataloader()  used for learning rate finding.
#' @param steps (integer) The number of steps to iterate over in the learning rate finder. Default: 100.
#' @param start_lr (float) The largest learning rate. Default: 1e-1.
#' @param end_lr (float) The lowest learning rate. Default: 1e-7.
#' @param plot (bool) A logical indicator whether to print a plot of learnign rate vs. loss.
#'
#' @return A dataframe with two columns: learning rate and loss
#' @export
lr_finder <- function(object, data, steps = 100, start_lr = 1e-1, end_lr = 1e-7, plot = TRUE) {
  # adjust batch size so that the steps number adds to one batch
  new_bs <- data$dataset$.length() / steps
  data$batch_sampler$batch_size <- new_bs

  scheduler <- luz_callback_lr_scheduler(
    lr_anneal,
    verbose=FALSE,
    start_lr = start_lr,
    end_lr = end_lr,
    n_iters = steps,
    call_on="on_train_batch_begin"
  )

  lr_profiler <- luz_callback_record_lr()

  fitted <- object %>%
    set_opt_hparams(lr = start_lr) %>%
    fit(data,
        epochs = 1,
        callbacks = list(scheduler, lr_profiler))

  lr_records <- data.frame(sapply(fitted$ctx$records$lr_finder, c))

  if (plot) {
    p <- lr_records %>%
      ggplot2::ggplot(aes(x = lr, y = loss)) +
        ggplot2::geom_line() +
        ggplot2::scale_x_log10() +
        ggplot2::xlab("Learning Rate") +
        ggplot2::ylab("Loss")

    print(p)
  }

  lr_records

}
