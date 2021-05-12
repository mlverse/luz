# early stopping

    Code
      expect_message({
        output <- mod %>% set_hparams(input_size = 10, output_size = 1) %>% fit(dl,
          verbose = TRUE, epochs = 25, callbacks = list(luz_callback_early_stopping(
            monitor = "train_loss", patience = 1)))
      })
    Message <message>
      Train metrics: Loss: 1.5301
    Message <message>
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
    Message <message>
      Epoch 2/25
    Message <message>
      Train metrics: Loss: 1.3641
    Message <message>
      Epoch 3/25
    Message <message>
      Train metrics: Loss: 1.2073
    Message <message>
      Epoch 4/25
    Message <message>
      Train metrics: Loss: 1.2524
    Message <message>
      Epoch 5/25
    Message <message>
      Train metrics: Loss: 1.1891
    Message <message>
      Early stopping at epoch 5 of 25

