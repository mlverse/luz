test_that("lr_finder works", {

  dl <- get_dl()
  model <- get_model()
  model <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam
    ) %>%
    set_hparams(input_size = 10, output_size = 1)

  records <- lr_finder(model, dl, verbose = FALSE)
  expect_s3_class(records, "lr_records")
  expect_s3_class(records, "data.frame")
  expect_equal(nrow(records), 100)

  expect_output(print(records))

  p <- plot(records)
  expect_s3_class(p, "gg")
})
