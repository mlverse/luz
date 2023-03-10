test_that("Metrics are correctly saved", {

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

  tmp <- tempfile()

  fitted <- model %>% fit(list(x, y), valid_data = 0.2, callbacks = list(
    luz_callback_tfevents(tmp, histograms = TRUE)
  ), verbose = 0)

  scalars <- tfevents::collect_events(tmp, type="scalar")
  expect_equal(nrow(scalars), 40)

  summaries <- tfevents::collect_events(tmp, type = "summary")
  histograms <- summaries[summaries$plugin == "histograms",]

  expect_equal(nrow(histograms), 20)
})
