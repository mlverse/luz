# early stopping

    Code
      expect_message({
        output <- mod %>% set_hparams(input_size = 10, output_size = 1) %>% fit(dl,
          verbose = TRUE, epochs = 25, callbacks = list(luz_callback_early_stopping(
            monitor = "train_loss", patience = 1)))
      })
    Message <message>
      Train metrics: Loss: 1.6053
      Early stopping at epoch 1 of 25

---

    Code
      expect_message({
        output <- mod %>% set_hparams(input_size = 10, output_size = 1) %>% fit(dl,
          verbose = TRUE, epochs = 25, callbacks = list(luz_callback_early_stopping(
            monitor = "train_loss", patience = 5, baseline = 0.001)))
      })
    Message <message>
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
            monitor = "train_mae", patience = 2, baseline = 0.91)))
      })
    Message <message>
      Train metrics: Loss: 1.2671 - MAE: 0.9163
      Epoch 2/25
      Train metrics: Loss: 1.2387 - MAE: 0.9058
      Epoch 3/25
      Train metrics: Loss: 1.212 - MAE: 0.8961
      Epoch 4/25
      Train metrics: Loss: 1.1866 - MAE: 0.8879
      Epoch 5/25
      Train metrics: Loss: 1.1625 - MAE: 0.8798
      Epoch 6/25
      Train metrics: Loss: 1.1396 - MAE: 0.8719
      Epoch 7/25
      Train metrics: Loss: 1.1179 - MAE: 0.8643
      Epoch 8/25
      Train metrics: Loss: 1.0973 - MAE: 0.8581
      Epoch 9/25
      Train metrics: Loss: 1.0778 - MAE: 0.8524
      Epoch 10/25
      Train metrics: Loss: 1.0593 - MAE: 0.847
      Epoch 11/25
      Train metrics: Loss: 1.0418 - MAE: 0.8417
      Epoch 12/25
      Train metrics: Loss: 1.0252 - MAE: 0.8365
      Epoch 13/25
      Train metrics: Loss: 1.0095 - MAE: 0.8314
      Epoch 14/25
      Train metrics: Loss: 0.9946 - MAE: 0.8265
      Epoch 15/25
      Train metrics: Loss: 0.9805 - MAE: 0.8216
      Epoch 16/25
      Train metrics: Loss: 0.9672 - MAE: 0.8168
      Epoch 17/25
      Train metrics: Loss: 0.9545 - MAE: 0.8122
      Epoch 18/25
      Train metrics: Loss: 0.9426 - MAE: 0.8077
      Epoch 19/25
      Train metrics: Loss: 0.9312 - MAE: 0.8035
      Epoch 20/25
      Train metrics: Loss: 0.9205 - MAE: 0.7994
      Epoch 21/25
      Train metrics: Loss: 0.9104 - MAE: 0.7953
      Epoch 22/25
      Train metrics: Loss: 0.9008 - MAE: 0.7914
      Epoch 23/25
      Train metrics: Loss: 0.8917 - MAE: 0.7875
      Epoch 24/25
      Train metrics: Loss: 0.8831 - MAE: 0.7838
      Epoch 25/25
      Train metrics: Loss: 0.8749 - MAE: 0.7801

# callback lr scheduler

    Code
      expect_message({
        output <- mod %>% set_hparams(input_size = 10, output_size = 1) %>% fit(dl,
          verbose = FALSE, epochs = 5, callbacks = list(luz_callback_lr_scheduler(
            torch::lr_multiplicative, verbose = TRUE, lr_lambda = function(epoch) 0.5)))
      })
    Message <message>
      Adjusting learning rate of group 1 to 0.0005
      Adjusting learning rate of group 1 to 0.0003
      Adjusting learning rate of group 1 to 0.0001
      Adjusting learning rate of group 1 to 0.0001
      Adjusting learning rate of group 1 to 0.0000

