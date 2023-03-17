# early stopping

    Code
      expect_message({
        output <- mod %>% set_hparams(input_size = 10, output_size = 1) %>% fit(dl,
          verbose = TRUE, epochs = 25, callbacks = list(luz_callback_early_stopping(
            monitor = "train_loss", patience = 1, min_delta = 0.02)))
      })
    Message
      Train metrics: Loss: 1.6053
      Epoch 2/25
      Train metrics: Loss: 1.5625
      Epoch 3/25
      Train metrics: Loss: 1.5222
      Epoch 4/25
      Train metrics: Loss: 1.4836
      Epoch 5/25
      Train metrics: Loss: 1.4466
      Epoch 6/25
      Train metrics: Loss: 1.4113
      Epoch 7/25
      Train metrics: Loss: 1.3775
      Epoch 8/25
      Train metrics: Loss: 1.3452
      Epoch 9/25
      Train metrics: Loss: 1.3144
      Epoch 10/25
      Train metrics: Loss: 1.285
      Epoch 11/25
      Train metrics: Loss: 1.257
      Epoch 12/25
      Train metrics: Loss: 1.2303
      Epoch 13/25
      Train metrics: Loss: 1.2049
      Epoch 14/25
      Train metrics: Loss: 1.1807
      Epoch 15/25
      Train metrics: Loss: 1.1576
      Epoch 16/25
      Train metrics: Loss: 1.1357
      Epoch 17/25
      Train metrics: Loss: 1.1149
      Epoch 18/25
      Train metrics: Loss: 1.0951
      Early stopping at epoch 18 of 25

---

    Code
      expect_message({
        output <- mod %>% set_hparams(input_size = 10, output_size = 1) %>% fit(dl,
          verbose = TRUE, epochs = 25, callbacks = list(luz_callback_early_stopping(
            monitor = "train_loss", patience = 5, baseline = 0.001)))
      })
    Message
      Train metrics: Loss: 1.279
      Epoch 2/25
      Train metrics: Loss: 1.2552
      Epoch 3/25
      Train metrics: Loss: 1.2335
      Epoch 4/25
      Train metrics: Loss: 1.2127
      Epoch 5/25
      Train metrics: Loss: 1.1927
      Early stopping at epoch 5 of 25

---

    Code
      expect_message({
        output <- mod %>% set_hparams(input_size = 10, output_size = 1) %>% fit(dl,
          verbose = TRUE, epochs = 25, callbacks = list(luz_callback_early_stopping(
            monitor = "train_mae", patience = 2, baseline = 0.91, min_delta = 0.01)))
      })
    Message
      Train metrics: Loss: 1.2671 - MAE: 0.9163
      Epoch 2/25
      Train metrics: Loss: 1.2387 - MAE: 0.9058
      Early stopping at epoch 2 of 25

