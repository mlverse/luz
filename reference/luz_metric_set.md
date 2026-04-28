# Creates a metric set

A metric set can be used to specify metrics that are only evaluated
during training, validation or both.

## Usage

``` r
luz_metric_set(metrics = NULL, train_metrics = NULL, valid_metrics = NULL)
```

## Arguments

- metrics:

  A list of luz_metrics that are meant to be used in both training and
  validation.

- train_metrics:

  A list of luz_metrics that are only used during training.

- valid_metrics:

  A list of luz_metrics that are only sued for validation.
