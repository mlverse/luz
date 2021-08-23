test_that("plot works", {

  set.seed(1)
  torch::torch_manual_seed(1)

  model <- get_model()
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam,
      metrics = list(
        luz_metric_rmse()
      )
    )

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    set_opt_hparams(lr = 0.001) %>%
    fit(dl, epochs = 10, verbose = FALSE, valid_data = dl)

  vdiffr::expect_doppelganger("ggplot2 histogram", plot(output))

})
