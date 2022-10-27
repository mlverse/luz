interrupt <- luz_callback(
  "interrupt",
  weight = Inf,
  failed = FALSE,
  on_epoch_end = function() {
    if (ctx$epoch == 5 && !self$failed) {
      self$failed <- TRUE
      self$metrics <- ctx$get_metrics_df()
      stop("Error on epoch 5")
    }
  }
)

clone_tensors <- function(x) {
  if (is.list(x))
    lapply(x, clone_tensors)
  else if (inherits(x, "torch_tensor"))
    x$clone()
  else
    x
}

track_weights <- luz_callback(
  "track_weights",
  weights = list(),
  opt = list(),
  initialize = function(on_end = TRUE) {
    self$on_end <- on_end
  },
  on_epoch_begin = function() {
    self$weights[[ctx$epoch]] <- lapply(ctx$model$state_dict(), function(x) x$clone())
    self$opt[[ctx$epoch]] <- clone_tensors(lapply(ctx$optimizers, function(opt) opt$state_dict()))
  },
  on_epoch_end = function() {
    # this actually is only called when no saved model exists, otherwise
    # the epoch is skipped by the autoresume callback.
    if (self$on_end) {
      self$on_epoch_begin()
    }
  }
)

test_that("resume a simple model", {

  x <- torch_randn(1000, 10)
  y <- torch_randn(1000, 1)

  model <- nn_linear %>%
    setup(optimizer = optim_sgd, loss = nnf_mse_loss) %>%
    set_hparams(in_features = 10, out_features = 1) %>%
    set_opt_hparams(lr = 0.01)

  temp <- tempfile()
  autoresume <- luz_callback_auto_resume(path = temp)
  inter <- interrupt()
  tr_w <- track_weights()

  # simulate an error during training
  expect_error(regexp = "Error on", {
    results <- model %>% fit(
      list(x, y),
      callbacks = list(tr_w, autoresume, inter),
      verbose = FALSE
    )
  })

  tr_w_resume <- track_weights()
  # reruning, now making sure no error will happen
  results_resume <- model %>% fit(
    list(x, y),
    callbacks = list(tr_w_resume, autoresume, inter),
    verbose = FALSE
  )

  metrics <- get_metrics(results_resume)

  expect_true(nrow(metrics) == 10)
  expect_true(all.equal(metrics[1:5,], inter$metrics))

  # expect that the first five weights are identical to the last one from
  # the first run.
  for(i in 1:5) {
    expect_true(torch_allclose(tr_w$weights[[5]]$weight, tr_w_resume$weights[[i]]$weight))
    expect_true(torch_allclose(tr_w$weights[[5]]$bias, tr_w_resume$weights[[i]]$bias))

    expect_identical(tr_w$opt[[i]], tr_w_resume$opt[[i]])
  }

  # Now that the run is complete, rerunning will trigger a completely new run.
  results_resume2 <- model %>% fit(
    list(x, y),
    callbacks = list(autoresume),
    verbose = FALSE
  )

  # we expect no identical metrics at all.
  expect_true(!identical(get_metrics(results_resume2), metrics))
})

test_that("resume a model with more than one optimizer", {

  x <- torch_randn(1000, 10)
  y <- torch_randn(1000, 1)

  module <- nn_module(
    inherit = nn_linear,
    set_optimizers = function(lr = 2*1e-4, betas = c(0.5, 0.999)) {
      list(
        weight = optim_adam(list(super$parameters$weight), lr = lr, betas = betas),
        bias = optim_adam(list(super$parameters$bias), lr = lr, betas = betas)
      )
    }
  )

  model <- module %>%
    setup(loss = nnf_mse_loss) %>%
    set_hparams(in_features = 10, out_features = 1) %>%
    set_opt_hparams(lr = 1e-4)

  temp <- tempfile()
  autoresume <- luz_callback_auto_resume(path = temp)
  tr_w <- track_weights()
  inter <- interrupt()

  # simulate an error during training
  expect_error(regexp = "Error on", {
    results <- model %>% fit(
      list(x, y),
      callbacks = list(tr_w, autoresume, inter),
      verbose = FALSE
    )
  })

  tr_w_resume <- track_weights()
  results_resume <- model %>% fit(
    list(x, y),
    callbacks = list(tr_w_resume, autoresume, inter),
    verbose = FALSE
  )

  for (i in 1:5) {
    expect_recursive_equal(tr_w$opt[[5]], tr_w_resume$opt[[1]])
  }

})

test_that("resume a model with learning rate scheduler", {
  cb_with_state <- luz_callback(
    weight = Inf,
    initialize = function() {
      self$i <- 1
    },
    on_epoch_end = function() {
      self$i <- self$i + 1
    },
    state_dict = function() {
      list(i = self$i)
    },
    load_state_dict = function(d) {
      self$i <- d$i
    }
  )


  x <- torch_randn(1000, 10)
  y <- torch_randn(1000, 1)

  model <- nn_linear %>%
    setup(optimizer = optim_sgd, loss = nnf_mse_loss) %>%
    set_hparams(in_features = 10, out_features = 1) %>%
    set_opt_hparams(lr = 0.01)

  temp <- tempfile()
  autoresume <- luz_callback_auto_resume(path = temp)
  inter <- interrupt()
  cb_state <- cb_with_state()

  # simulate an error during training
  expect_error(regexp = "Error on", {
    results <- model %>% fit(
      list(x, y),
      callbacks = list(autoresume, cb_state, inter),
      verbose = FALSE
    )
  })

  cb_state2 <- cb_with_state()
  results_resume <- model %>% fit(
    list(x, y),
    callbacks = list(autoresume, cb_state2, inter),
    verbose = FALSE
  )

  # we would expect a larger number if the state is not correctly recovered
  expect_equal(cb_state2$i, 10)
  expect_equal(cb_state$i, 6)
})


test_that("resume works when model has been explicitly interrupted", {
  # sometimes we want to early stop, in this case we need to make sure that
  # this interruptions doesn't count as 'not finished training'.

  x <- torch_randn(1000, 10)
  y <- torch_randn(1000, 1)

  model <- nn_linear %>%
    setup(optimizer = optim_sgd, loss = nnf_mse_loss) %>%
    set_hparams(in_features = 10, out_features = 1) %>%
    set_opt_hparams(lr = 0.01)

  temp <- tempfile()
  autoresume <- luz_callback_auto_resume(path = temp)
  early_stop <- luz_callback_early_stopping(monitor = "train_loss", patience = 1)

  results <- model %>% fit(
    list(x, y),
    callbacks = list(autoresume, early_stop),
    verbose = FALSE,
    epochs = 100
  )

  results2 <- model %>% fit(
    list(x, y),
    callbacks = list(autoresume, early_stop),
    verbose = FALSE,
    epochs = 100
  )

  # values would be identical if results2 was resumed from results1
  expect_true(get_metrics(results2)$value[1] != get_metrics(results)$value[1])
})

test_that("can use the resume_from callback", {

  x <- torch_randn(1000, 10)
  y <- torch_randn(1000, 1)

  model <- nn_linear %>%
    setup(optimizer = optim_sgd, loss = nnf_mse_loss) %>%
    set_hparams(in_features = 10, out_features = 1) %>%
    set_opt_hparams(lr = 0.01)

  temp <- tempfile()
  checkpoint <- luz_callback_model_checkpoint(
    path = temp,
    monitor = "train_loss"
  )

  tr <- track_weights()
  result <- model %>% fit(
    list(x, y),
    callbacks = list(tr, checkpoint),
    verbose = FALSE
  )

  tr2 <- track_weights(on_end = FALSE)
  resume_from <- luz_callback_resume_from_checkpoint(path = temp)
  result2 <- model %>% fit(
    list(x, y),
    callbacks = list(tr2, resume_from),
    verbose = FALSE
  )

  expect_recursive_equal(
    tr$weights[[10]],
    tr2$weights[[1]]
  )
})
