# resuming a model with a lr scheduler callback is correct

    Code
      result <- model %>% fit(list(x, y), callbacks = list(autoresume,
        luz_callback_lr_scheduler(lr_step, step_size = 1L),
        luz_callback_simulate_failure(at_epoch = 11L), luz_callback_lr_progress()),
      verbose = FALSE)
    Message
      lr=1e-06
      lr=1e-06
      lr=1e-06
      lr=1e-06
      lr=1e-06
      lr=1e-07
      lr=1e-08
      lr=1e-09
      lr=1e-10
      lr=1e-11

