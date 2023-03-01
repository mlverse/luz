# get a nice error message when metric fails updating

    Error while calling callback with class <metrics_callback/LuzCallback/R6> at on_train_batch_end.
    Caused by error in `FUN()`:
    ! Error when evaluating update for metric with abbrev "h" and class <hello/LuzMetric/R6>
    i The error happened at iter 1 of epoch 1.
    i The model was in training mode.
    Caused by error in `metric$update()`:
    ! error in metric!

# error gracefully on failed `compute`.

    Error while calling callback with class <metrics_callback/LuzCallback/R6> at on_train_end.
    Caused by error in `self$call_compute_on_metric()`:
    ! Error when evaluating compute for metric with abbrev "h" and class <hello/LuzMetric/R6>
    i The error happened at iter 31 of epoch 1.
    i The model was in training mode.
    Caused by error in `metric$compute()`:
    ! Error computing metric.

