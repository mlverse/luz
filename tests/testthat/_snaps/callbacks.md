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

---

    Code
      expect_message({
        output <- mod %>% set_hparams(input_size = 10, output_size = 1) %>% fit(dl,
          verbose = TRUE, epochs = 25, callbacks = list(luz_callback_early_stopping(
            monitor = "train_mae", patience = 5, baseline = 0.85)))
      })
    Message <message>
      Train metrics: Loss: 1.475 - MAE: 0.9769
      Epoch 2/25
      Train metrics: Loss: 1.1386 - MAE: 0.8167
      Epoch 3/25
      Train metrics: Loss: 1.2144 - MAE: 0.871
      Epoch 4/25
      Train metrics: Loss: 1.0531 - MAE: 0.8151
      Epoch 5/25
      Train metrics: Loss: 1.2934 - MAE: 0.9233
      Epoch 6/25
      Train metrics: Loss: 1.047 - MAE: 0.7962
      Epoch 7/25
      Train metrics: Loss: 1.1791 - MAE: 0.8182
      Epoch 8/25
      Train metrics: Loss: 1.2227 - MAE: 0.9059
      Epoch 9/25
      Train metrics: Loss: 1.3843 - MAE: 0.9421
      Epoch 10/25
      Train metrics: Loss: 1.1501 - MAE: 0.8601
      Epoch 11/25
      Train metrics: Loss: 0.8833 - MAE: 0.755
      Epoch 12/25
      Train metrics: Loss: 1.0919 - MAE: 0.8362
      Epoch 13/25
      Train metrics: Loss: 1.5323 - MAE: 0.9833
      Epoch 14/25
      Train metrics: Loss: 1.2334 - MAE: 0.9069
      Epoch 15/25
      Train metrics: Loss: 0.8962 - MAE: 0.7398
      Epoch 16/25
      Train metrics: Loss: 1.0139 - MAE: 0.843
      Epoch 17/25
      Train metrics: Loss: 1.0814 - MAE: 0.8298
      Epoch 18/25
      Train metrics: Loss: 1.1125 - MAE: 0.8586
      Epoch 19/25
      Train metrics: Loss: 1.1964 - MAE: 0.8567
      Epoch 20/25
      Train metrics: Loss: 0.8883 - MAE: 0.7459
      Early stopping at epoch 20 of 25

