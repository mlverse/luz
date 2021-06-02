test_that("callback interrupt works", {

  model <- get_model()
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam,
    )

  hello <- 0
  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, verbose = FALSE, epochs = 25, callbacks = list(
      luz_callback(on_epoch_begin = function() {
        if (ctx$epoch == 3)
          rlang::interrupt()
      })(),
      luz_callback(on_interrupt = function() {
        hello <<- 1
      })()
    ))

  expect_equal(length(output$records$metrics$train), 2)
  expect_equal(hello, 1)

})
