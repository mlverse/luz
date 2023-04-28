test_that("Can use mixed precision callback", {

  x <- torch_randn(1000, 10, device=if(torch::cuda_is_available()) "cuda" else "cpu")
  y <- torch_randn(1000, 1)

  module <- nn_linear

  model <- module %>%
    setup(loss = nnf_mse_loss, optimizer = optim_adam) %>%
    set_hparams(in_features = 10, out_features = 1) %>%
    set_opt_hparams(lr = 1e-4)

  callback_for_testing <- luz_callback(
    on_fit_begin = function() {
      expect_true(!identical(ctx$step_opt, default_step_opt))
    },
    on_train_batch_begin = function() {
      if (ctx$iter == 1 && ctx$epoch == 1) {
	      y <- torch_matmul(x, x$t())
      	if (torch::cuda_is_available())
  	      expect_equal(y$dtype$.type(), "Half")
     	  else
  	      expect_equal(y$dtype$.type(), "BFloat16")
      }
    },
    on_train_batch_before_backward = function() {
      if (ctx$iter == 1 && ctx$epoch == 1) {
        y <- torch_matmul(x, x$t())
        expect_equal(y$dtype$.type(), "Float")
      }
    }
  )

  fitted <- model %>% fit(list(x, y), valid_data = 0.2, callbacks = list(
    luz_callback_mixed_precision(enabled = cuda_is_available()),
    callback_for_testing()
  ), accelerator = accelerator(cpu = !cuda_is_available()), verbose = FALSE)

})
