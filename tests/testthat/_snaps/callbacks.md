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
      Epoch 2/25
    Message <message>
      Train metrics: Loss: 1.654
    Message <message>
      Early stopping at epoch 2 of 25

