# Create a new callback

Create a new callback

## Usage

``` r
luz_callback(
  name = NULL,
  ...,
  private = NULL,
  active = NULL,
  parent_env = parent.frame(),
  inherit = NULL
)
```

## Arguments

- name:

  name of the callback

- ...:

  Public methods of the callback. The name of the methods is used to
  know how they should be called. See the details section.

- private:

  An optional list of private members, which can be functions and
  non-functions.

- active:

  An optional list of active binding functions.

- parent_env:

  An environment to use as the parent of newly-created objects.

- inherit:

  A R6ClassGenerator object to inherit from; in other words, a
  superclass. This is captured as an unevaluated expression which is
  evaluated in `parent_env` each time an object is instantiated.

## Value

A `luz_callback` that can be passed to
[`fit.luz_module_generator()`](https://mlverse.github.io/luz/reference/fit.luz_module_generator.md).

## Details

Let’s implement a callback that prints ‘Iteration `n`’ (where `n` is the
iteration number) for every batch in the training set and ‘Done’ when an
epoch is finished. For that task we use the `luz_callback` function:

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

`luz_callback()` takes named functions as `...` arguments, where the
name indicates the moment at which the callback should be called. For
instance `on_train_batch_end()` is called for every batch at the end of
the training procedure, and `on_epoch_end()` is called at the end of
every epoch.

The returned value of `luz_callback()` is a function that initializes an
instance of the callback. Callbacks can have initialization parameters,
like the name of a file where you want to log the results. In that case,
you can pass an `initialize` method when creating the callback
definition, and save these parameters to the `self` object. In the above
example, the callback has a `message` parameter that is printed at the
end of each epoch.

Once a callback is defined it can be passed to the `fit` function via
the `callbacks` parameter:

    fitted <- net %>%
      setup(...) %>%
      fit(..., callbacks = list(
        print_callback(message = "Done!")
      ))

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

## Prediction callbacks

You can also use callbacks when using
[`predict()`](https://rdrr.io/r/stats/predict.html). In this case the
supported callback methods are detailed below:

    Start predict
     - on_predict_begin
     Start prediction loop
      - on_predict_batch_begin
      - on_predict_batch_end
     End prediction loop
     - on_predict_end
    End predict

## Evaluate callbacks

Callbacks can also be used with
[`evaluate()`](https://mlverse.github.io/luz/reference/evaluate.md), in
this case, the callbacks that are used are equivalent to those of the
validation loop when using
[`fit()`](https://generics.r-lib.org/reference/fit.html):

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

## See also

Other luz_callbacks:
[`luz_callback_auto_resume()`](https://mlverse.github.io/luz/reference/luz_callback_auto_resume.md),
[`luz_callback_csv_logger()`](https://mlverse.github.io/luz/reference/luz_callback_csv_logger.md),
[`luz_callback_early_stopping()`](https://mlverse.github.io/luz/reference/luz_callback_early_stopping.md),
[`luz_callback_interrupt()`](https://mlverse.github.io/luz/reference/luz_callback_interrupt.md),
[`luz_callback_keep_best_model()`](https://mlverse.github.io/luz/reference/luz_callback_keep_best_model.md),
[`luz_callback_lr_scheduler()`](https://mlverse.github.io/luz/reference/luz_callback_lr_scheduler.md),
[`luz_callback_metrics()`](https://mlverse.github.io/luz/reference/luz_callback_metrics.md),
[`luz_callback_mixed_precision()`](https://mlverse.github.io/luz/reference/luz_callback_mixed_precision.md),
[`luz_callback_mixup()`](https://mlverse.github.io/luz/reference/luz_callback_mixup.md),
[`luz_callback_model_checkpoint()`](https://mlverse.github.io/luz/reference/luz_callback_model_checkpoint.md),
[`luz_callback_profile()`](https://mlverse.github.io/luz/reference/luz_callback_profile.md),
[`luz_callback_progress()`](https://mlverse.github.io/luz/reference/luz_callback_progress.md),
[`luz_callback_resume_from_checkpoint()`](https://mlverse.github.io/luz/reference/luz_callback_resume_from_checkpoint.md),
[`luz_callback_train_valid()`](https://mlverse.github.io/luz/reference/luz_callback_train_valid.md)

## Examples

``` r
print_callback <- luz_callback(
 name = "print_callback",
 on_train_batch_end = function() {
   cat("Iteration ", ctx$iter, "\n")
 },
 on_epoch_end = function() {
   cat("Done!\n")
 }
)
```
