# Get started with luz

``` r
library(luz)
library(torch)
```

Luz is a high-level API for torch that aims to encapsulate the
**training loop** into a set of reusable pieces of code. Luz reduces the
boilerplate code required to train a model with torch and avoids the
error prone `zero_grad()` - `backward()` -
[`step()`](https://rdrr.io/r/stats/step.html) sequence of calls, and
also simplifies the process of moving data and models between CPUs and
GPUs. Luz is designed to be highly flexible by providing a layered API
that allows it to be useful no matter the level of control you need for
your training loop.

Luz is heavily inspired by other higher level frameworks for deep
learning, to cite a few:

- [FastAI](https://docs.fast.ai/): we are heavily inspired by the FastAI
  library, especially the `Learner` object and the callbacks API.

- [Keras](https://keras.io/): We are also heavily inspired by Keras,
  especially callback names. The lightning module interface is similar
  to `compile`, too.

- [PyTorch Lightning](https://lightning.ai/pages/open-source/): The idea
  of the `luz_module` being a subclass of `nn_module` is inspired by the
  **`LightningModule`** object in lightning.

- [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/): The
  internal device placement API is heavily inspired by Accelerate, but
  is much more modest in features. Currently only CPU and Single GPU are
  supported.

## Training a `nn_module`

As much as possible, luz tries to reuse the existing structures from
torch. A model in luz is defined identically as you would define it if
using raw torch. For a specific example, this is the definition of a
feed-forward CNN that can be used to classify digits from the MNIST
dataset:

``` r
net <- nn_module(
  "Net",
  initialize = function(num_class) {
    self$conv1 <- nn_conv2d(1, 32, 3, 1)
    self$conv2 <- nn_conv2d(32, 64, 3, 1)
    self$dropout1 <- nn_dropout2d(0.25)
    self$dropout2 <- nn_dropout2d(0.5)
    self$fc1 <- nn_linear(9216, 128)
    self$fc2 <- nn_linear(128, num_class)
  },
  forward = function(x) {
    x <- self$conv1(x)
    x <- nnf_relu(x)
    x <- self$conv2(x)
    x <- nnf_relu(x)
    x <- nnf_max_pool2d(x, 2)
    x <- self$dropout1(x)
    x <- torch_flatten(x, start_dim = 2)
    x <- self$fc1(x)
    x <- nnf_relu(x)
    x <- self$dropout2(x)
    x <- self$fc2(x)
    x
  }
)
```

We can now train this model in the `train_dl` and validate it in the
`test_dl` `torch::dataloaders()` with:

``` r
fitted <- net %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_accuracy
    )
  ) %>%
  set_hparams(num_class = 10) %>% 
  set_opt_hparams(lr = 0.003) %>% 
  fit(train_dl, epochs = 10, valid_data = test_dl)
```

Let’s understand what happens in this chunk of code:

1.  The `setup` function allows you to configure the loss (objective)
    function and the optimizer that you will use to train your model.
    Optionally you can pass a list of metrics that are tracked during
    the training procedure. **Note:** the loss function can be any
    function taking `input` and `target` tensors and returning a scalar
    tensor value, and the optimizer can be any core torch optimizer or
    custom ones created with the
    [`torch::optimizer()`](https://torch.mlverse.org/docs/reference/optimizer.html)
    function.
2.  The
    [`set_hparams()`](https://mlverse.github.io/luz/reference/set_hparams.md)
    function allows you to set hyper-parameters that should be passed to
    the module `initialize()` method. For example in this case we pass
    `num_classes = 10`.
3.  The
    [`set_opt_hparams()`](https://mlverse.github.io/luz/reference/set_opt_hparams.md)
    function allows you to pass hyper-parameters that are used by the
    optimizer function. For example,
    [`optim_adam()`](https://torch.mlverse.org/docs/reference/optim_adam.html)
    can take the `lr` parameter specifying the learning rate and we
    specify it with `lr = 0.003`.
4.  The `fit` method will take the model specification provided by
    [`setup()`](https://mlverse.github.io/luz/reference/setup.md) and
    run the training procedure using the specified training and
    validation `torch::dataloaders()` as well as the number of epochs.
    **Note:** we again reuse core torch data structures, instead of
    providing our own data loading functionality.
5.  The returned object `fitted` contains the trained model as well as
    the record of metrics and losses produced during training. It can
    also be used for producing predictions and for evaluating the
    trained model on other datasets.

When fitting, luz will use the fastest possible accelerator; if a
CUDA-capable GPU is available it will be used, otherwise we fall back to
the CPU. It also automatically moves data, optimizers, and models to the
selected device so you don’t need to handle it manually (which is in
general very error prone).

To create predictions from the trained model you can use the `predict`
method:

``` r
predictions <- predict(fitted, test_dl)
```

## The training loop

You now have a general idea of how to use the `fit` function and now
it’s important to have an overview of what’s happening inside it. In
pseudocode, here’s what `fit` does. This is not fully detailed but
should help you to build your intuition:

``` r
# -> Initialize objects: model, optimizers.
# -> Select fitting device.
# -> Move data, model, optimizers to the selected device.
# -> Start training
for (epoch in 1:epochs) {
  # -> Training procedure
  for (batch in train_dl) {
    # -> Calculate model `forward` method.
    # -> Calculate the loss
    # -> Update weights
    # -> Update metrics and tracking loss
  }
  # -> Validation procedure
  for (batch in valid_dl) {
    # -> Calculate model `forward` method.
    # -> Calculate the loss
    # -> Update metrics and tracking loss
  }
}
# -> End training
```

## Metrics

One of the most important parts in machine learning projects is choosing
the evaluation metric. Luz allows tracking many different metrics during
training with minimal code changes.

In order to track metrics, you only need to modify the `metrics`
parameter in the `setup` function:

``` r
fitted <- net %>%
  setup(
    ...
    metrics = list(
      luz_metric_accuracy
    )
  ) %>%
  fit(...)
```

Luz provides implementations of a few of the most used metrics. If a
metric is not available you can always implement a new one using the
`luz_metric` function.

In order to implement a new `luz_metric` we need to implement 3 methods:

- `initialize`: defines the metric initial state. This function is
  called for each epoch for both training and validation loops.

- `update`: updates the metric internal state. This function is called
  at every training and validation step with the predictions obtained by
  the model and the target values obtained from the dataloader.

- `compute`: uses the internal state to compute metric values. This
  function is called whenever we need to obtain the current metric
  value. Eg, it’s called every training step for metrics displayed in
  the progress bar, but only called once per epoch to record it’s value
  when the progress bar is not displayed.

Optionally, you can implement an `abbrev` field that gives the metric an
abbreviation that will be used when displaying metric information in the
console or tracking record. If no `abbrev` is passed, the class name
will be used.

Let’s take a look at the implementation of `luz_metric_accuracy` so you
can see how to implement a new one:

``` r
luz_metric_accuracy <- luz_metric(
  # An abbreviation to be shown in progress bars, or 
  # when printing progress
  abbrev = "Acc", 
  # Initial setup for the metric. Metrics are initialized
  # every epoch, for both training and validation
  initialize = function() {
    self$correct <- 0
    self$total <- 0
  },
  # Run at every training or validation step and updates
  # the internal state. The update function takes `preds`
  # and `target` as parameters.
  update = function(preds, target) {
    pred <- torch::torch_argmax(preds, dim = 2)
    self$correct <- self$correct + (pred == target)$
      to(dtype = torch::torch_float())$
      sum()$
      item()
    self$total <- self$total + pred$numel()
  },
  # Use the internal state to query the metric value
  compute = function() {
    self$correct/self$total
  }
)
```

**Note**: It’s good practice that the `compute` metric returns regular R
values instead of torch tensors and other parts of luz will expect that.

## Evaluate

Once a model has been trained you might want to evaluate its performance
on a different dataset. For that reason, luz provides the
[`?evaluate`](https://mlverse.github.io/luz/reference/evaluate.md)
function that takes a fitted model and a dataset and computes the
metrics attached to the model.

Evaluate returns a `luz_module_evaluation` object that you can query for
metrics using the `get_metrics` function or simply `print` to see the
results.

For example:

``` r
evaluation <- fitted %>% evaluate(data = valid_dl)
metrics <- get_metrics(evaluation)
print(evaluation)
```

    #> A `luz_module_evaluation`
    #> -- Results ---------------------------------------------------------------------
    #> loss: 1.8892
    #> mae: 1.0522
    #> mse: 1.645
    #> rmse: 1.2826

## Customizing with callbacks

Luz provides different ways to customize the training progress depending
on the level of control you need in the training loop. The fastest way
and the more ‘reusable’, in the sense that you can create training
modifications that can be used in many different situations, is via
**callbacks**.

The training loop in luz has many *breakpoints* that can call arbitrary
R functions. This functionality allows you to customize the training
process without having to modify the general training logic.

Luz implements 3 default callbacks that occur in every training
procedure:

- **train-eval callback**: Sets the model to `train()` or
  [`eval()`](https://rdrr.io/r/base/eval.html) depending on if the
  procedure is doing training or validation.

- **metrics callback**: evaluate metrics during training and validation
  process.

- **progress callback**: implements a progress bar and prints progress
  information during training.

You can also implement custom callbacks that modify or act specifically
for your training procedure. For example:

Let’s implement a callback that prints ‘Iteration `n`’ (where `n` is the
iteration number) for every batch in the training set and ‘Done’ when an
epoch is finished. For that task we use the `luz_callback` function:

``` r
print_callback <- luz_callback(
  name = "print_callback",
  initialize = function(message) {
    self$message <- message
  },
  on_train_batch_end = function() {
    cat("Iteration ", ctx$iter, "\n")
  },
  on_epoch_end = function() {
    cat(self$message, "\n")
  }
)
```

[`luz_callback()`](https://mlverse.github.io/luz/reference/luz_callback.md)
takes named functions as `...` arguments, where the name indicates the
moment at which the callback should be called. For instance
`on_train_batch_end()` is called for every batch at the end of the
training procedure, and `on_epoch_end()` is called at the end of every
epoch.

The returned value of
[`luz_callback()`](https://mlverse.github.io/luz/reference/luz_callback.md)
is a function that initializes an instance of the callback. Callbacks
can have initialization parameters, like the name of a file where you
want to log the results. In that case, you can pass an `initialize`
method when creating the callback definition, and save these parameters
to the `self` object. In the above example, the callback has a `message`
parameter that is printed at the end of each epoch.

Once a callback is defined it can be passed to the `fit` function via
the `callbacks` parameter:

``` r
fitted <- net %>%
  setup(...) %>%
  fit(..., callbacks = list(
    print_callback(message = "Done!")
  ))
```

Callbacks can be called in many different positions of the training
loop, including combinations of them. Here’s an overview of possible
callback *breakpoints*:

    Start Fit
       - on_fit_begin
      Start Epoch Loop
         - on_epoch_begin
        Start Train
           - on_train_begin
          Start Batch Loop
             - on_train_batch_begin
              Start Default Training Step
                - on_train_batch_after_pred
                - on_train_batch_after_loss
                - on_train_batch_before_backward
                - on_train_batch_before_step
                - on_train_batch_after_step
              End Default Training Step:
             - on_train_batch_end
          End Batch Loop
           - on_train_end
        End Train
        Start Valid
           - on_valid_begin
          Start Batch Loop
             - on_valid_batch_begin
              Start Default Validation Step
                - on_valid_batch_after_pred
                - on_valid_batch_after_loss
              End Default Validation Step
             - on_valid_batch_end
          End Batch Loop
           - on_valid_end
        End Valid
          - on_epoch_end
      End Epoch Loop
       - on_fit_end
    End Fit

Every step marked with `on_*` is a point in the training procedure that
is available for callbacks to be called.

The other important part of callbacks is the `ctx` (context) object. See
[`help("ctx")`](https://mlverse.github.io/luz/reference/ctx.md) for
details.

By default, callbacks are called in the same order as they were passed
to `fit` (or `predict` or `evaluate`), but you can provide a `weight`
attribute that will control the order in which it will be called. For
example, if one callback has `weight = 10` and another has `weight = 1`,
then the first one is called after the second one. Callbacks that don’t
specify a `weight` attribute are considered `weight = 0`. A few built-in
callbacks in luz already provide a weight value. For example, the
[`?luz_callback_early_stopping`](https://mlverse.github.io/luz/reference/luz_callback_early_stopping.md)
has a weight of `Inf`, since in general we want to run it as the last
thing in the loop.

The `ctx` object is used in luz to share information between the
training loop and callbacks, model methods, and metrics. The table below
describes information available in the `ctx` by default. Other callbacks
could potentially modify these attributes or add new ones.

| Attribute        | Description                                                                                                                                                                                                                                                                                                                                                                               |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `verbose`        | The value (`TRUE` or `FALSE`) attributed to the `verbose` argument in `fit` .                                                                                                                                                                                                                                                                                                             |
| `accelerator`    | Accelerator object used to query the correct device to place models, data, etc. It assumes the value passed to the `accelerator` parameter in `fit`.                                                                                                                                                                                                                                      |
| `model`          | Initialized `nn_module` object that will be trained during the `fit` procedure.                                                                                                                                                                                                                                                                                                           |
| `optimizers`     | A named list of optimizers used during training.                                                                                                                                                                                                                                                                                                                                          |
| `data`           | The currently in-use dataloader. When training it’s `ctx$train_data`, when doing validation its `ctx$valid_data`. It can also be the prediction dataset when in `predict`.                                                                                                                                                                                                                |
| `train_data`     | Dataloader passed to the `data` argument in `fit`. Modified to yield data in the selected device.                                                                                                                                                                                                                                                                                         |
| `valid_data`     | Dataloader passed to the `valid_data` argument in `fit`. Modified to yield data in the selected device.                                                                                                                                                                                                                                                                                   |
| `min_epochs`     | Minimum number of epochs the model will be trained for.                                                                                                                                                                                                                                                                                                                                   |
| `max_epochs`     | Maximum number of epochs the model will be trained for.                                                                                                                                                                                                                                                                                                                                   |
| `epoch`          | Current training epoch.                                                                                                                                                                                                                                                                                                                                                                   |
| `iter`           | Current training iteration. It’s reset every epoch and when going from training to validation.                                                                                                                                                                                                                                                                                            |
| `training`       | Whether the model is in training or validation mode. See also [`help("luz_callback_train_valid")`](https://mlverse.github.io/luz/reference/luz_callback_train_valid.md)                                                                                                                                                                                                                   |
| `callbacks`      | List of callbacks that will be called during the training procedure. It’s the union of the list passed to the `callbacks` parameter and the default `callbacks`.                                                                                                                                                                                                                          |
| `step`           | Closure that will be used to do one `step` of the model. It’s used for both training and validation. Takes no argument, but can access the `ctx` object.                                                                                                                                                                                                                                  |
| `call_callbacks` | Call callbacks by name. For example `call_callbacks("on_train_begin")` will call all callbacks that provide methods for this point.                                                                                                                                                                                                                                                       |
| `batch`          | Last batch obtained by the dataloader. A batch is a [`list()`](https://rdrr.io/r/base/list.html) with 2 elements, one that is used as `input` and the other as `target`.                                                                                                                                                                                                                  |
| `input`          | First element of the last batch obtained by the current dataloader.                                                                                                                                                                                                                                                                                                                       |
| `target`         | Second element of the last batch obtained by the current dataloader.                                                                                                                                                                                                                                                                                                                      |
| `pred`           | Last predictions obtained by `ctx$model$forward` . **Note:** can be potentially modified by previously ran callbacks. Also note that this might not be available if you used a custom training step.                                                                                                                                                                                      |
| `loss_fn`        | The active loss function that will be minimized during training.                                                                                                                                                                                                                                                                                                                          |
| `loss`           | Last computed loss from the model. **Note:** this might not be available if you modified the training or validation step.                                                                                                                                                                                                                                                                 |
| `opt`            | Current optimizer, ie. the optimizer that will be used to do the next `step` to update parameters.                                                                                                                                                                                                                                                                                        |
| `opt_nm`         | Current optimizer name. By default it’s `opt` , but can change if your model uses more than one optimizer depending on the set of parameters being optimized.                                                                                                                                                                                                                             |
| `metrics`        | [`list()`](https://rdrr.io/r/base/list.html) with current metric objects that are `update`d at every `on_train_batch_end()` or `on_valid_batch_end()`. See also [`help("luz_callback_metrics")`](https://mlverse.github.io/luz/reference/luz_callback_metrics.md)                                                                                                                         |
| `records`        | [`list()`](https://rdrr.io/r/base/list.html) recording metric values for training and validation for each epoch. See also [`help("luz_callback_metrics")`](https://mlverse.github.io/luz/reference/luz_callback_metrics.md) . Also records profiling metrics. See [`help("luz_callback_profile")`](https://mlverse.github.io/luz/reference/luz_callback_profile.md) for more information. |
| `handlers`       | A named [`list()`](https://rdrr.io/r/base/list.html) of handlers that is passed to [`rlang::with_handlers()`](https://rlang.r-lib.org/reference/with_handlers.html) during the training loop and can be used to handle errors or conditions that might be raised by other callbacks.                                                                                                      |
| `epoch_handlers` | A named list of handlers that is used with [`rlang::with_handlers()`](https://rlang.r-lib.org/reference/with_handlers.html). Those handlers are used inside the epochs loop, thus you can handle epoch specific conditions, that won’t necessarily end training.                                                                                                                          |

Context attributes

Attributes in `ctx` can be used to produce the desired behavior of
callbacks. You can find information about the context object using
[`help("ctx")`](https://mlverse.github.io/luz/reference/ctx.md). In our
example, we use the `ctx$iter` attribute to print the iteration number
for each training batch.

## Next steps

In this article you learned how to train your first model using luz and
the basics of customization using both custom metrics and callbacks.

Luz also allows more flexible modifications of the training loop
described in
[`vignette("custom-loop")`](https://mlverse.github.io/luz/articles/custom-loop.md).

You should now be able to follow the examples marked with the ‘basic’
category in the [examples
gallery](https://mlverse.github.io/luz/articles/examples/index.html).
