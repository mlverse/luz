# get a nice error message when metric fails updating

    Error when evaluating update for metric with abbrev "h" and class <hello/LuzMetric/R6>
    i The error happened at iter 1 of epoch 1.
    i The model was in training mode.
    Caused by error in `metric$update()`:
    ! error in metric!

# error gracefully on failed `compute`.

    Error when evaluating compute for metric with abbrev "h" and class <hello/LuzMetric/R6>
    i The error happened at iter 32 of epoch 1.
    i The model was in training mode.
    Caused by error in `metric$compute()`:
    ! Error computing metric.

