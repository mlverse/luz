# luz (development version)

# luz 0.5.0

* Added mixed precision callback. (#127)
* Added support for torch iterable datasets. (#135)
* Fixed a bug when trying to resume models trained with learning rate schedulers. (#137)
* Added support for learning rate schedulers that take the current loss as arguments. (#140)
* Added French translation of luz messages. (@cregouby #148)

# luz 0.4.0

## Breaking changes

* `drop_last=TRUE` is now the default for training dataloaders created by luz (when eg. you pass a list or a torch dataset as data input) (#117)
* The default profile callback no longer tracks intra step timings as it adds a non ignorable overhead. (#125)

## New features

* Added support for arm Mac's and the MPS device. (#104)
* Refactor checkpointing in luz - we now also serialize optimizer state and callbacks state. (#107)
* Added a `luz_callback_autoresume()` allowing to easily resume training runs that might have crashed. (#107)
* Added the `luz_callback_resume_from_checkpoint()` allowing one to resume a training run from a checkpoint file. (#107)
* Users can now chose if metrics should be called on both training and validation,
only training or only validation. See `luz_metric_set()` for more information. (#112)
* Improved how errors raised on user code, eg while calling metrics or callbacks
are raised. This helps a lot when debuging errors in callbacks and metrics. (#112)
* `loss_fn` is now a field of the context, thus callbacks can override it when needed. (#112)
* `luz_callback_mixup` now supports the `run_valid` and `auto_loss` arguments. (#112)
* `ctx` now aliases to the default `opt` and `opt_name` when a single optimizer is specified (ie. most cases) (#114)
* Added `tfevents` callback for logging the loss and getting weights histograms. (#118)
* You can now specify metrics to be evaluated during `evaluate`. (#123)

## Bug fixes

* Bug fix: `accelerator`s `cpu` argument is always respected. (#119)
* Handled `rlang` and `ggplot2` deprecations. (#120)
* Better handling of metrics environments.
* Faster garbage collection of dataloaders iterators, so we use less memory. (#122)
* Much faster loss averaging at every step. Can have hight influence in training times for large number of iterations per epoch. (#124)

# luz 0.3.1

* Re-submission to fix vignette rendering.

# luz 0.3.0

## Breaking changes

* `lr_finder()` now by default divides the range between `start_lr` and `end_lr` into log-spaced intervals, following the fast.ai implementation. Cf. Sylvain Gugger's post: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html. The previous behavior can be achieved passing `log_spaced_intervals=FALSE` to the function. (#82, @skeydan)
* `plot.lr_records()` now in addition plots an exponentially weighted moving average of the loss (again, see Sylvain Gugger's post), with a weighting coefficient of `0.9` (which seems a reasonable value for the default setting of 100 learning-rate-incrementing intervals). (#82, @skeydan)

## Documentation

* Many wording improvements in the getting started guides (#81 #94, @jonthegeek).

## New features

* Added MixUp callback and helper loss function and functional logic. (#82, @skeydan).
* Added a `luz_callback_gradient_clip` inspired by FastAI's implementation. (#90)
* Added a `backward` argument to `setup` allowing one to customize how `backward` is called for the loss scalar value. (#93)
* Added the `luz_callback_keep_best_model()` to reload the weights from the best model after training is finished. (#95)

# luz 0.2.0

## New features

* Allow users to provide the minimum and maximum number of epochs when calling `fit.luz_module_generator()`. Removed `ctx$epochs` from context object and replaced it with `ctx$min_epochs` and `ctx$max_epochs` (#53, @mattwarkentin).
* Early stopping will now only occur if the minimum number of training epochs has been met (#53, @mattwarkentin).
* Added `cuda_index` argument to `accelerator` to allow selecting an specific GPU when multiple are present (#58, @cmcmaster1).
* Implemented `lr_finder` (#59, @cmcmaster1).
* We now handle different kinds of data arguments passed to `fit` using the `as_dataloader()` method (#66).
* `valid_data` can now be scalar value indicating the proportion of `data` that will be used for fitting. This only works if `data` is a torch dataset or a list. (#69)
* You can now supply `dataloader_options` to `fit` to pass additional information to `as_dataloader()`. (#71)
* Implemented the `evaluate` function allowing users to get metrics from a model in a new dataset. (#73)

## Bug fixes

* Fixed bug in CSV logger callback that was saving the logs as a space delimited file (#52, @mattwarkentin).
* Fixed bug in the length of the progress bar for the validation dataset (#52, @mattwarkentin).
* Fixed bugs in early stopping callback related to them not working properly when `patience = 1` and when they are specified before other logging callbacks. (#76)

## Internal changes

* `ctx$data` now refers to the current in use `data` instead of always refering to `ctx$train_data`. (#54)
* Refactored the `ctx` object to make it safer and avoid returing it in the output. (#73)

# luz 0.1.0

* Added a `NEWS.md` file to track changes to the package.
