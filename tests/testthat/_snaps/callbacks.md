# early stopping

    Code
      expect_message({
        output <- mod %>% set_hparams(input_size = 10, output_size = 1) %>% fit(dl,
          verbose = TRUE, epochs = 25, callbacks = list(luz_callback_early_stopping(
            monitor = "train_loss", patience = 1)))
      })
    Message <message>
      Train metrics: Loss: 1.5301
      Early stopping at epoch 1 of 25

---

    Code
      expect_message({
        output <- mod %>% set_hparams(input_size = 10, output_size = 1) %>% fit(dl,
          verbose = TRUE, epochs = 25, callbacks = list(luz_callback_early_stopping(
            monitor = "train_loss", patience = 5, baseline = 0.001)))
      })
    Message <message>
      Train metrics: Loss: 1.4807
      Epoch 2/25
      Train metrics: Loss: 1.3641
      Epoch 3/25
      Train metrics: Loss: 1.2073
      Epoch 4/25
      Train metrics: Loss: 1.2524
      Epoch 5/25
      Train metrics: Loss: 1.1891
      Early stopping at epoch 5 of 25

