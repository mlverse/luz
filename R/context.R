#' Context object
#'
#' Context objects used in luz to share information between model methods,
#' metrics and callbacks.
#'
#' @includeRmd man/rmd/ctx.Rmd details
#' @rdname ctx
#' @name ctx
#'
#' @seealso Context object: [context]
NULL

#' Context object
#'
#' @description
#' Context object storing information about the model training context.
#' See also [ctx].
#'
#' @param name name of the metric
#' @param value value to log
#' @param what (string) What you are logging.
#' @param set (string) Usually 'train' or 'valid' indicating the set you want
#'  to log to. But can be arbitrary info.
#' @param value Arbitrary value to log.
#' @param index Index that this value should be logged. If `NULL` the value
#'  is added to the end of list, otherwise the index is used.
#' @param append If `TRUE` and a value in the corresponding index already
#'  exists, then value is appended to the current value. If `FALSE` value
#'  is overwritten in favor of the new value.
#' @param epoch The epoch you want to extract metrics from.
#' @param verbose Whether the context should be in verbose mode or not.
#' @param accelerator A luz [accelerator()] that configures device placement and
#'   others.
#' @param callbacks A list of callbacks used by the model. See [luz_callback()].
#' @param training A boolean that indicates if the context is in training mode or not.
#' @param records New set of records to be set.
#'
context <- R6::R6Class(
  "luz_context",
  lock_objects = TRUE,
  public = list(

    #' @description
    #' Initializes the context object with minimal necessary information.
    initialize = function(verbose, accelerator, callbacks, training) {
      self$set_verbose(verbose)
      self$accelerator <- accelerator %||% accelerator()
      self$callbacks <- initialize_callbacks(callbacks, self)
      self$training <- training
    },

    #' @field buffers This is a list of buffers that callbacks can use to write temporary
    #'   information into `ctx`.
    buffers = list(),

    #' @description
    #' Allows logging arbitrary information in the `ctx`.
    log = function(what, set, value, index = NULL, append = TRUE) {

      if (is.null(index)) {
        index <- length(private$.records[[what]][[set]]) + 1L
      }

      current <- if (append) {
        if (length(private$.records[[what]][[set]]) < index) {
          NULL
        } else {
          private$.records[[what]][[set]][[index]]
        }
      } else {
        NULL
      }

      value <- append(current, value)

      if (is.null(private$.records[[what]]))
        private$.records[[what]][[set]] <- list()

      private$.records[[what]][[set]][[index]] <- value
      invisible(self)
    },
    #' @description
    #' Log a metric gen its name and value.
    #' Metric values are indexed by epoch.
    log_metric = function(name, value) {
      set <- if (self$training) "train" else "valid"

      value <- list(value)
      names(value) <- name

      self$log("metrics", set, value, index = self$epoch)


      invisible(self)
    },
    #' @description
    #' Get a specific value from the log.
    get_log = function(what, set, index = NULL) {
      get_log(self, what = what, set = set, index = index)
    },
    #' @description
    #' Get all metric given an epoch and set.
    get_metrics = function(set, epoch = NULL) {
      get_all_metrics(self, set = set, epoch = epoch)
    },
    #' @description
    #' Get the value of a metric given its name, epoch and set.
    get_metric = function(name, set, epoch= NULL) {
      get_metric(self, name = name, set = set, epoch = epoch)
    },
    #' @description
    #' Get formatted metrics values
    get_formatted_metrics = function(set, epoch = NULL) {
      get_formatted_metrics(self, set = set, epoch = epoch)
    },
    #' @description
    #' Get a data.frame containing all metrics.
    get_metrics_df = function() {
      get_metrics(self)
    },
    #' @description Allows setting the `verbose` attribute.
    #' @param verbose boolean. If `TRUE` verbose mode is used. If `FALSE` non verbose.
    #'   if `NULL` we use the result of [interactive()].
    set_verbose = function(verbose = NULL) {
      if (is.null(verbose)) {
        private$.verbose <- interactive()
      } else {
        private$.verbose <- verbose
      }
    },
    #' @description Removes unnecessary information from the context object.
    clean = function() {
      lapply(FUN = function(x) private[[x]] <- NULL, c(
        ".callbacks",
        ".train_data",
        ".valid_data",
        ".accelerator",
        ".optimizers",
        ".verbose",
        ".handlers",
        ".epoch_handlers",
        ".metrics",
        ".training",
        ".batch",
        ".iter",
        ".pred",
        ".opt",
        ".opt_name",
        ".data",
        ".loss_fn",
        ".loss",
        ".loss_grad",
        ".epoch"
      ))
      self$buffers <- NULL
    },
    #' @description
    #' Call the selected callbacks. Where `name` is the callback types to call, eg
    #' 'on_epoch_begin'.
    call_callbacks = function(name) {
      call_all_callbacks(self$callbacks, name)
    },
    #' @description
    #' Returns a list containing minimal information from the context. Used to
    #' create the returned values.
    state_dict = function() {
      output <- list(
        model = self$model,
        records = self$records,
        ctx = list(
          hparams = self$hparams,
          opt_hparams = self$opt_hparams
        )
      )
      # Remove the context reference so the context can be correctly
      # deleted.
      bind_context(output$model, NULL)
      output
    },
    #' @description
    #' Are you sure you know what you are doing?
    unsafe_set_records = function(records) {
      if (!length(private$.records$metrics$train) == 0) {
        rlang::warn("You are unsafe setting records and it's overriding current data.")
      }
      private$.records <- records
    }
  ),
  active = list(
    #' @field records stores information about values logged with `self$log`.
    records = function(x) {
      if (!missing(x))
        rlang::abort("Not allowed to modify records manually. Use ctx$log() or ctx$log_metric()")

      private$.records
    },
    #' @field device allows querying the current accelerator device
    device = function(x) {

      if (!missing(x))
        rlang::abort("Not allowed to modify the device manually. Modify the ctx$accelerator")

      if (is.null(self$accelerator))
        rlang::abort("Context doesn't have an accelerator attached.")

      self$accelerator$device
    },
    #' @field callbacks list of callbacks that will be called.
    callbacks = function(new) {
      if(missing(new))
        return(private$.callbacks)
      private$.callbacks <- ctx_check_callbacks(new)
      invisible(private$.callbacks)
    },
    #' @field iter current iteration
    iter = function(new) {
      if (missing(new))
        return(private$.iter)
      private$.iter <- ctx_check_iter(new)
      invisible(private$.iter)
    },
    #' @field batch the current batch data. a list with input data and targets.
    batch = function(new) {
      if (missing(new))
        return(private$.batch)

      private$.batch <- new
    },
    #' @field input a shortcut for `ctx$batch[[1]]`
    input = function(new) {
      if (missing(new))
        return(private$.batch[[1]])

      private$.batch[[1]] <- new
    },
    #' @field target a shortcut for `ctx$batch[[2]]`
    target = function(new) {
      if (missing(new))
        return(private$.batch[[2]])

      private$.batch[[2]] <- new
    },
    #' @field min_epochs the minimum number of epochs that the model will run on.
    min_epochs = function(new) {
      if (missing(new))
        return(private$.epochs$min_epochs)
      ctx_check_epochs(new, self$max_epochs)
      private$.epochs$min_epochs <- new
    },
    #' @field max_epochs the maximum number of epochs that the model will run.
    max_epochs = function(new) {
      if (missing(new))
        return(private$.epochs$max_epochs)
      ctx_check_epochs(self$min_epochs, new)
      private$.epochs$max_epochs <- new
    },
    #' @field hparams a list of hyperparameters that were used to initialize `ctx$model`.
    hparams = function(new) {
      if (missing(new))
        return(private$.hparams)
      private$.hparams <- new
    },
    #' @field opt_hparams a list of hyperparameters used to initialize the `ctx$optimizers`.
    opt_hparams = function(new) {
      if (missing(new))
        return(private$.opt_hparams)
      private$.opt_hparams <- new
    },
    #' @field train_data a dataloader that is used for training the model
    train_data = function(new) {
      if (missing(new))
        return(private$.train_data)
      private$.train_data <- new
    },
    #' @field valid_data a dataloader using during model validation
    valid_data = function(new) {
      if (missing(new))
        return(private$.valid_data)
      private$.valid_data <- new
    },
    #' @field accelerator an [accelerator()] used to move data, model and etc the the correct
    #'   device.
    accelerator = function(new) {
      if (missing(new))
        return(private$.accelerator)
      private$.accelerator <- new
    },
    #' @field optimizers a named list of optimizers that will be used during model training.
    optimizers = function(new) {
      if (missing(new))
        return(private$.optimizers)
      private$.optimizers <- ctx_check_optimizers(new)
    },
    #' @field verbose bool wether the process is in verbose mode or not.
    verbose = function(new) {
      if (missing(new))
        return(private$.verbose)
      self$set_verbose(new)
    },
    #' @field handlers List of error handlers that can be used. See [rlang::try_fetch()]
    #'   for more info.
    handlers = function(new) {
      if (missing(new))
        return(private$.handlers)
      private$.handlers <- new
    },
    #' @field epoch_handlers List of error handlers that can be used. See [rlang::try_fetch()]
    #'   for more info.
    epoch_handlers = function(new) {
      if (missing(new))
        return(private$.epoch_handlers)
      private$.epoch_handlers <- new
    },
    #' @field training A bool indicating if the model is in training or validation mode.
    training = function(new){
      if (missing(new))
        return(private$.training)
      private$.training <- new
    },
    #' @field model The model being trained.
    model = function(new) {
      if (missing(new))
        return(private$.model)
      private$.model <- new
      bind_context(private$.model, self)
    },
    #' @field pred Last predicted values.
    pred = function(new) {
      if (missing(new))
        return(private$.pred)
      private$.pred <- new
    },
    #' @field opt Current optimizer.
    opt = function(new) {
      if (missing(new)) {
        if (!is.null(private$.opt)) {
          return(private$.opt)
        } else {
          if (length(self$optimizers) == 1) {
            return(self$optimizers[[1]])
          }
        }
        cli::cli_abort("{.var ctx$opt} not set.")
      }
      private$.opt <- new
    },
    #' @field opt_name Current optimizer name.
    opt_name = function(new) {
      if (missing(new)) {
        if (!is.null(private$.opt_name)) {
          return(private$.opt_name)
        } else {
          if (length(self$optimizers) == 1) {
            return(names(self$optimizers))
          }
        }
        cli::cli_abort("{.var ctx$opt_name} not set.")
      }
      private$.opt_name <- new
    },
    #' @field data Current dataloader in use.
    data = function(new) {
      if (missing(new))
        return(private$.data)
      private$.data <- new
    },
    #' @field loss_fn Loss function used to train the model
    loss_fn = function(new) {
      if (missing(new))
        return(private$.loss_fn)
      private$.loss_fn <- new
    },
    #' @field loss Last computed loss values. Detached from the graph.
    loss = function(new) {
      if (missing(new))
        return(private$.loss)
      private$.loss <- new
    },
    #' @field loss_grad Last computed loss value, not detached, so you can do additional
    #'   tranformation.
    loss_grad = function(new) {
      if (missing(new))
        return(private$.loss_grad)
      private$.loss_grad <- new
    },
    #' @field epoch Current epoch.
    epoch = function(new) {
      if (missing(new))
        return(private$.epoch)
      private$.epoch <- new
    },
    #' @field metrics List of metrics that are tracked by the process.
    metrics = function(new) {
      if (missing(new))
        return(private$.metrics)
      private$.metrics <- new
    },
    #' @field step_opt Defines how step is called for the optimizer. It must be a function
    #' taking an optimizer as argument.
    step_opt = function(new) {
      if (missing(new)) return(private$.step_opt)
      private$.step_opt <- new
    }
  ),
  private = list(
    # Fields that make sense to be kept after the model has been trained.
    .records = list(metrics = list(
      train = list(),
      valid = list()
    )),
    .hparams = NULL,
    .opt_hparams = NULL,
    .epochs = list(min_epochs = 0, max_epochs = 999999999),
    .model = NULL,

    # These fields are used during training, prediction or evaluation, but they
    # are not important after the process has finished. They are likely to be
    # recreated for each process that happen on the model.
    .callbacks = NULL,
    .train_data = NULL,
    .valid_data = NULL,
    .accelerator = NULL,
    .optimizers = NULL,
    .verbose = NULL,
    .handlers = list(),
    .epoch_handlers = list(),
    .metrics = NULL,
    .step_opt = NULL,

    # Fields that are overwritten during model training. They are more or
    # less transient, and their values don't make sense after the model
    # has been trained.
    .training = NULL,
    .batch = NULL,
    .iter = NULL,
    .pred = list(),
    .opt = NULL,
    .opt_name = NULL,
    .data = NULL,
    .loss_fn = NULL,
    .loss = NULL,
    .loss_grad = NULL,
    .epoch = NULL
  )
)

fit_context <- R6::R6Class(
  classname = "luz_fit_context",
  inherit = context,
  public = list(
    initialize = function (verbose, accelerator, callbacks, module, hparams,
                           opt_hparams, data, valid_data, epochs, dataloader_options) {

      super$initialize(
        verbose = verbose,
        accelerator = accelerator,
        callbacks = append(default_callbacks(), callbacks),
        training = TRUE
      )

      self$hparams <- get_hparams(module) %||% list()
      self$opt_hparams <- get_opt_hparams(module) %||% list()

      self$model <- do.call(module, self$hparams)
      self$optimizers <- do.call(self$model$set_optimizers, self$opt_hparams)
      self$loss_fn <- self$model$loss
      self$step_opt <- default_step_opt

      if (rlang::is_scalar_double(valid_data)) {
        c(data, valid_data) %<-% create_valid_data(data, valid_data)
      }

      c(data, valid_data) %<-% apply_dataloader_options(data, valid_data, dataloader_options)

      c(model, optimizers, data, valid_data) %<-%
        self$accelerator$prepare(
          self$model,
          self$optimizers,
          data,
          valid_data
        )

      self$model <- model
      self$optimizers <- optimizers

      self$data <- data
      self$train_data <- data
      self$valid_data <- valid_data

      if (length(epochs) == 1) epochs <- c(0, epochs)
      self$min_epochs <- epochs[[1]]
      self$max_epochs <- epochs[[2]]

    }
  )
)

predict_context <- R6::R6Class(
  classname = "luz_predict_context",
  inherit = context,
  public = list(
    initialize = function(model, newdata, callbacks, accelerator, verbose,
                          dataloader_options, callbacks_default) {

      super$initialize(
        verbose = verbose,
        accelerator = accelerator,
        callbacks = c(callbacks_default(), callbacks),
        training = FALSE
      )

      c(., newdata) %<-% apply_dataloader_options(NULL, newdata, dataloader_options)
      c(model, data) %<-% self$accelerator$prepare(model, newdata)

      self$model <- model
      self$model$eval()

      self$data <- data
    }
  )
)

evaluate_context <- R6::R6Class(
  classname = "luz_evaluate_context",
  inherit = predict_context,
  public = list(
    initialize = function(..., opt_hparams) {
      super$initialize(...)
      self$epoch <- 1L
      self$opt_hparams <- opt_hparams
      # we actually only use the optimizer names ...
      self$optimizers <- do.call(self$model$set_optimizers, self$opt_hparams)
      # evaluate computes the loss function, and it's better to refer to it from
      # the context.
      self$loss_fn <- self$model$loss
    }
  )
)

make_metrics_df <- function(metrics_list, set) {
  purrr::imap_dfr(metrics_list, function(x, epoch) {
    purrr::imap_dfr(x, function(value, metric_name) {
      data.frame(
        stringsAsFactors = FALSE,
        set = set,
        metric = metric_name,
        epoch = epoch,
        value = value
      )
    })
  })
}

ctx_check_callbacks <- function(x) {
  for (i in seq_along(x)) {
    cb <- x[[i]]
    if (!inherits(cb, "LuzCallback")) {
      message <- "Expected a LuzCallback but got an object with class '{class(cb)[1]}' at index {i}."
      rlang::abort(glue::glue(message))
    }
  }
  x
}

ctx_check_iter <- function(x) {
  if (!rlang::is_scalar_integerish(x)) {
    message <- "Expected iter to be a scalar integer. Got {str(x)}."
    rlang::abort(glue::glue(x))
  }
  x
}

ctx_check_epochs <- function(min, max) {
  if (!rlang::is_scalar_integerish(min))
    rlang::abort("Expected `min_epochs` to be a scalar integer, got {str(min)}.")

  if (!rlang::is_scalar_integerish(max))
    rlang::abort("Expected `max_epochs` to be a scalar integer, got {str(max)}.")

  if (min > max)
    rlang::abort("`min_epochs` is higher than `max_epochs` and that's not allowed.")

  invisible(list(min, max))
}

ctx_check_optimizers <- function(new) {
  if (!is.list(new)) {
    new <- list(opt = new)
  }

  if (!rlang::is_named(new)) {
    rlang::abort(c("List of optimizers is not named.",
                   "When returning a list of optimizers, the list must be named."))
  }

  for (i in new) {
    if (!torch::is_optimizer(i))
      rlang::abort("Expected a torch optimizer but got an object with class '{class(i)[1]}'.")
  }

  invisible(new)
}

default_step_opt <- function(opt) {
  opt$step()
}
