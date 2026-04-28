# Package index

## Training

- [`setup()`](https://mlverse.github.io/luz/reference/setup.md) :

  Set's up a `nn_module` to use with luz

- [`fit(`*`<luz_module_generator>`*`)`](https://mlverse.github.io/luz/reference/fit.luz_module_generator.md)
  :

  Fit a `nn_module`

- [`predict(`*`<luz_module_fitted>`*`)`](https://mlverse.github.io/luz/reference/predict.luz_module_fitted.md)
  : Create predictions for a fitted model

- [`evaluate()`](https://mlverse.github.io/luz/reference/evaluate.md) :
  Evaluates a fitted model on a dataset

- [`set_hparams()`](https://mlverse.github.io/luz/reference/set_hparams.md)
  : Set hyper-parameter of a module

- [`set_opt_hparams()`](https://mlverse.github.io/luz/reference/set_opt_hparams.md)
  : Set optimizer hyper-parameters

- [`get_metrics()`](https://mlverse.github.io/luz/reference/get_metrics.md)
  : Get metrics from the object

- [`ctx`](https://mlverse.github.io/luz/reference/ctx.md) : Context
  object

- [`context`](https://mlverse.github.io/luz/reference/context.md) :
  Context object

- [`lr_finder()`](https://mlverse.github.io/luz/reference/lr_finder.md)
  : Learning Rate Finder

- [`as_dataloader()`](https://mlverse.github.io/luz/reference/as_dataloader.md)
  : Creates a dataloader from its input

## Metrics

- [`luz_metric()`](https://mlverse.github.io/luz/reference/luz_metric.md)
  : Creates a new luz metric
- [`luz_metric_accuracy()`](https://mlverse.github.io/luz/reference/luz_metric_accuracy.md)
  : Accuracy
- [`luz_metric_binary_accuracy()`](https://mlverse.github.io/luz/reference/luz_metric_binary_accuracy.md)
  : Binary accuracy
- [`luz_metric_binary_accuracy_with_logits()`](https://mlverse.github.io/luz/reference/luz_metric_binary_accuracy_with_logits.md)
  : Binary accuracy with logits
- [`luz_metric_binary_auroc()`](https://mlverse.github.io/luz/reference/luz_metric_binary_auroc.md)
  : Computes the area under the ROC
- [`luz_metric_mae()`](https://mlverse.github.io/luz/reference/luz_metric_mae.md)
  : Mean absolute error
- [`luz_metric_mse()`](https://mlverse.github.io/luz/reference/luz_metric_mse.md)
  : Mean squared error
- [`luz_metric_multiclass_auroc()`](https://mlverse.github.io/luz/reference/luz_metric_multiclass_auroc.md)
  : Computes the multi-class AUROC
- [`luz_metric_rmse()`](https://mlverse.github.io/luz/reference/luz_metric_rmse.md)
  : Root mean squared error
- [`luz_metric_set()`](https://mlverse.github.io/luz/reference/luz_metric_set.md)
  : Creates a metric set

## Misc

- [`nn_mixup_loss()`](https://mlverse.github.io/luz/reference/nn_mixup_loss.md)
  :

  Loss to be used with `callbacks_mixup()`.

- [`nnf_mixup()`](https://mlverse.github.io/luz/reference/nnf_mixup.md)
  : Mixup logic

## Callbacks

- [`luz_callback()`](https://mlverse.github.io/luz/reference/luz_callback.md)
  : Create a new callback
- [`luz_callback_auto_resume()`](https://mlverse.github.io/luz/reference/luz_callback_auto_resume.md)
  : Resume training callback
- [`luz_callback_csv_logger()`](https://mlverse.github.io/luz/reference/luz_callback_csv_logger.md)
  : CSV logger callback
- [`luz_callback_early_stopping()`](https://mlverse.github.io/luz/reference/luz_callback_early_stopping.md)
  : Early stopping callback
- [`luz_callback_gradient_clip()`](https://mlverse.github.io/luz/reference/luz_callback_gradient_clip.md)
  : Gradient clipping callback
- [`luz_callback_interrupt()`](https://mlverse.github.io/luz/reference/luz_callback_interrupt.md)
  : Interrupt callback
- [`luz_callback_keep_best_model()`](https://mlverse.github.io/luz/reference/luz_callback_keep_best_model.md)
  : Keep the best model
- [`luz_callback_lr_scheduler()`](https://mlverse.github.io/luz/reference/luz_callback_lr_scheduler.md)
  : Learning rate scheduler callback
- [`luz_callback_metrics()`](https://mlverse.github.io/luz/reference/luz_callback_metrics.md)
  : Metrics callback
- [`luz_callback_mixed_precision()`](https://mlverse.github.io/luz/reference/luz_callback_mixed_precision.md)
  : Automatic Mixed Precision callback
- [`luz_callback_mixup()`](https://mlverse.github.io/luz/reference/luz_callback_mixup.md)
  : Mixup callback
- [`luz_callback_model_checkpoint()`](https://mlverse.github.io/luz/reference/luz_callback_model_checkpoint.md)
  : Checkpoints model weights
- [`luz_callback_profile()`](https://mlverse.github.io/luz/reference/luz_callback_profile.md)
  : Profile callback
- [`luz_callback_progress()`](https://mlverse.github.io/luz/reference/luz_callback_progress.md)
  : Progress callback
- [`luz_callback_resume_from_checkpoint()`](https://mlverse.github.io/luz/reference/luz_callback_resume_from_checkpoint.md)
  : Allow resume model training from a specific checkpoint
- [`luz_callback_tfevents()`](https://mlverse.github.io/luz/reference/luz_callback_tfevents.md)
  : tfevents callback
- [`luz_callback_train_valid()`](https://mlverse.github.io/luz/reference/luz_callback_train_valid.md)
  : Train-eval callback

## Accelerator

- [`accelerator()`](https://mlverse.github.io/luz/reference/accelerator.md)
  : Create an accelerator

## Serialization

- [`luz_save()`](https://mlverse.github.io/luz/reference/luz_save.md) :
  Saves luz objects to disk
- [`luz_load()`](https://mlverse.github.io/luz/reference/luz_load.md) :
  Load trained model
- [`luz_load_model_weights()`](https://mlverse.github.io/luz/reference/luz_load_model_weights.md)
  [`luz_save_model_weights()`](https://mlverse.github.io/luz/reference/luz_load_model_weights.md)
  : Loads model weights into a fitted object.
- [`luz_load_checkpoint()`](https://mlverse.github.io/luz/reference/luz_load_checkpoint.md)
  : Loads a checkpoint
