# luz (development version)

* Fixed bug in CSV logger callback that was saving the logs as a space delimited file (#52, @mattwarkentin).
* Fixed bug in the length of the progress bar for the validation dataset (#52, @mattwarkentin).
* `ctx$data` now refers to the current in use `data` instead of always refering to `ctx$train_data`. (#54)
* Allow users to provide the minimum and maximum number of epochs when calling `fit.luz_module_generator()`. Removed `ctx$epochs` from context object and replaced it with `ctx$min_epochs` and `ctx$max_epochs` (#53, @mattwarkentin).
* Early stopping will now only occur if the minimum number of training epochs has been met (#53, @mattwarkentin).
* Added `cuda_index` argument to `accelerator` to allow selecting an specific GPU when multiple are present (#58, @cmcmaster1).
* Implemented `lr_finder` (#59, @cmcmaster1).
* We now handle different kinds of data arguments passed to `fit` using the `as_dataloader()` method (#66).
* `valid_data` can now be scalar value indicating the proportion of `data` that will be used for fitting. This only works if `data` is a torch dataset or a list. (#69)
* You can now supply `dataloader_options` to `fit` to pass additional information to `as_dataloader()`. (#71)
* Refactored the `ctx` object to make it safer and avoid returing it in the output. (#73)
* Implemented the `evaluate` function allowing users to get metrics from a model in a new datase. (#73)

# luz 0.1.0

* Added a `NEWS.md` file to track changes to the package.
