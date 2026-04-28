# Changelog

## luz 0.5.2

CRAN release: 2026-04-28

- New maintainer: Tomasz Kalinowski.
- Switched CI GPU runners to ephemeral runs-on instances.
- Bumped CI CUDA container to 12.8.1.
- Fixed bugs in example vignettes (`mnist-vae`, `text-generation`).

## luz 0.5.1

CRAN release: 2025-10-30

- fixed a bug preventing additional arguments passed to `predict` to be
  forwarded to `model$predict`
  ([\#157](https://github.com/mlverse/luz/issues/157))

## luz 0.5.0

CRAN release: 2025-07-29

- Added mixed precision callback.
  ([\#127](https://github.com/mlverse/luz/issues/127))
- Added support for torch iterable datasets.
  ([\#135](https://github.com/mlverse/luz/issues/135))
- Fixed a bug when trying to resume models trained with learning rate
  schedulers. ([\#137](https://github.com/mlverse/luz/issues/137))
- Added support for learning rate schedulers that take the current loss
  as arguments. ([\#140](https://github.com/mlverse/luz/issues/140))
- Added French translation of luz messages.
  ([@cregouby](https://github.com/cregouby)
  [\#148](https://github.com/mlverse/luz/issues/148))

## luz 0.4.0

CRAN release: 2023-04-17

### Breaking changes

- `drop_last=TRUE` is now the default for training dataloaders created
  by luz (when eg. you pass a list or a torch dataset as data input)
  ([\#117](https://github.com/mlverse/luz/issues/117))
- The default profile callback no longer tracks intra step timings as it
  adds a non ignorable overhead.
  ([\#125](https://github.com/mlverse/luz/issues/125))

### New features

- Added support for arm Mac’s and the MPS device.
  ([\#104](https://github.com/mlverse/luz/issues/104))
- Refactor checkpointing in luz - we now also serialize optimizer state
  and callbacks state.
  ([\#107](https://github.com/mlverse/luz/issues/107))
- Added a `luz_callback_autoresume()` allowing to easily resume training
  runs that might have crashed.
  ([\#107](https://github.com/mlverse/luz/issues/107))
- Added the
  [`luz_callback_resume_from_checkpoint()`](https://mlverse.github.io/luz/reference/luz_callback_resume_from_checkpoint.md)
  allowing one to resume a training run from a checkpoint file.
  ([\#107](https://github.com/mlverse/luz/issues/107))
- Users can now chose if metrics should be called on both training and
  validation, only training or only validation. See
  [`luz_metric_set()`](https://mlverse.github.io/luz/reference/luz_metric_set.md)
  for more information.
  ([\#112](https://github.com/mlverse/luz/issues/112))
- Improved how errors raised on user code, eg while calling metrics or
  callbacks are raised. This helps a lot when debuging errors in
  callbacks and metrics.
  ([\#112](https://github.com/mlverse/luz/issues/112))
- `loss_fn` is now a field of the context, thus callbacks can override
  it when needed. ([\#112](https://github.com/mlverse/luz/issues/112))
- `luz_callback_mixup` now supports the `run_valid` and `auto_loss`
  arguments. ([\#112](https://github.com/mlverse/luz/issues/112))
- `ctx` now aliases to the default `opt` and `opt_name` when a single
  optimizer is specified (ie. most cases)
  ([\#114](https://github.com/mlverse/luz/issues/114))
- Added `tfevents` callback for logging the loss and getting weights
  histograms. ([\#118](https://github.com/mlverse/luz/issues/118))
- You can now specify metrics to be evaluated during `evaluate`.
  ([\#123](https://github.com/mlverse/luz/issues/123))

### Bug fixes

- Bug fix: `accelerator`s `cpu` argument is always respected.
  ([\#119](https://github.com/mlverse/luz/issues/119))
- Handled `rlang` and `ggplot2` deprecations.
  ([\#120](https://github.com/mlverse/luz/issues/120))
- Better handling of metrics environments.
- Faster garbage collection of dataloaders iterators, so we use less
  memory. ([\#122](https://github.com/mlverse/luz/issues/122))
- Much faster loss averaging at every step. Can have hight influence in
  training times for large number of iterations per epoch.
  ([\#124](https://github.com/mlverse/luz/issues/124))

## luz 0.3.1

CRAN release: 2022-09-06

- Re-submission to fix vignette rendering.

## luz 0.3.0

CRAN release: 2022-08-19

### Breaking changes

- [`lr_finder()`](https://mlverse.github.io/luz/reference/lr_finder.md)
  now by default divides the range between `start_lr` and `end_lr` into
  log-spaced intervals, following the fast.ai implementation. Cf.
  Sylvain Gugger’s post:
  <https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html>.
  The previous behavior can be achieved passing
  `log_spaced_intervals=FALSE` to the function.
  ([\#82](https://github.com/mlverse/luz/issues/82),
  [@skeydan](https://github.com/skeydan))
- `plot.lr_records()` now in addition plots an exponentially weighted
  moving average of the loss (again, see Sylvain Gugger’s post), with a
  weighting coefficient of `0.9` (which seems a reasonable value for the
  default setting of 100 learning-rate-incrementing intervals).
  ([\#82](https://github.com/mlverse/luz/issues/82),
  [@skeydan](https://github.com/skeydan))

### Documentation

- Many wording improvements in the getting started guides
  ([\#81](https://github.com/mlverse/luz/issues/81)
  [\#94](https://github.com/mlverse/luz/issues/94),
  [@jonthegeek](https://github.com/jonthegeek)).

### New features

- Added MixUp callback and helper loss function and functional logic.
  ([\#82](https://github.com/mlverse/luz/issues/82),
  [@skeydan](https://github.com/skeydan)).
- Added a `luz_callback_gradient_clip` inspired by FastAI’s
  implementation. ([\#90](https://github.com/mlverse/luz/issues/90))
- Added a `backward` argument to `setup` allowing one to customize how
  `backward` is called for the loss scalar value.
  ([\#93](https://github.com/mlverse/luz/issues/93))
- Added the
  [`luz_callback_keep_best_model()`](https://mlverse.github.io/luz/reference/luz_callback_keep_best_model.md)
  to reload the weights from the best model after training is finished.
  ([\#95](https://github.com/mlverse/luz/issues/95))

## luz 0.2.0

CRAN release: 2021-10-07

### New features

- Allow users to provide the minimum and maximum number of epochs when
  calling
  [`fit.luz_module_generator()`](https://mlverse.github.io/luz/reference/fit.luz_module_generator.md).
  Removed `ctx$epochs` from context object and replaced it with
  `ctx$min_epochs` and `ctx$max_epochs`
  ([\#53](https://github.com/mlverse/luz/issues/53),
  [@mattwarkentin](https://github.com/mattwarkentin)).
- Early stopping will now only occur if the minimum number of training
  epochs has been met ([\#53](https://github.com/mlverse/luz/issues/53),
  [@mattwarkentin](https://github.com/mattwarkentin)).
- Added `cuda_index` argument to `accelerator` to allow selecting an
  specific GPU when multiple are present
  ([\#58](https://github.com/mlverse/luz/issues/58),
  [@cmcmaster1](https://github.com/cmcmaster1)).
- Implemented `lr_finder`
  ([\#59](https://github.com/mlverse/luz/issues/59),
  [@cmcmaster1](https://github.com/cmcmaster1)).
- We now handle different kinds of data arguments passed to `fit` using
  the
  [`as_dataloader()`](https://mlverse.github.io/luz/reference/as_dataloader.md)
  method ([\#66](https://github.com/mlverse/luz/issues/66)).
- `valid_data` can now be scalar value indicating the proportion of
  `data` that will be used for fitting. This only works if `data` is a
  torch dataset or a list.
  ([\#69](https://github.com/mlverse/luz/issues/69))
- You can now supply `dataloader_options` to `fit` to pass additional
  information to
  [`as_dataloader()`](https://mlverse.github.io/luz/reference/as_dataloader.md).
  ([\#71](https://github.com/mlverse/luz/issues/71))
- Implemented the `evaluate` function allowing users to get metrics from
  a model in a new dataset.
  ([\#73](https://github.com/mlverse/luz/issues/73))

### Bug fixes

- Fixed bug in CSV logger callback that was saving the logs as a space
  delimited file ([\#52](https://github.com/mlverse/luz/issues/52),
  [@mattwarkentin](https://github.com/mattwarkentin)).
- Fixed bug in the length of the progress bar for the validation dataset
  ([\#52](https://github.com/mlverse/luz/issues/52),
  [@mattwarkentin](https://github.com/mattwarkentin)).
- Fixed bugs in early stopping callback related to them not working
  properly when `patience = 1` and when they are specified before other
  logging callbacks. ([\#76](https://github.com/mlverse/luz/issues/76))

### Internal changes

- `ctx$data` now refers to the current in use `data` instead of always
  refering to `ctx$train_data`.
  ([\#54](https://github.com/mlverse/luz/issues/54))
- Refactored the `ctx` object to make it safer and avoid returing it in
  the output. ([\#73](https://github.com/mlverse/luz/issues/73))

## luz 0.1.0

CRAN release: 2021-06-17

- Added a `NEWS.md` file to track changes to the package.
