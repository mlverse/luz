LuzMetric <- R6::R6Class(
  "LuzMetric",
  lock_objects = FALSE,
  public = list(
    format = function(v) {
      if (is.numeric(v))
        round(v, 4)
      else if (is.list(v)) {
        v <- lapply(v, round, 4)
        paste0(glue::glue("{names(v)}: {v}"), collapse = " | ")
      }
    }
  )
)

luz_metric <- function(name = NULL, ..., public, active, parent_env = parent.frame()) {
  public <- rlang::list2(...)
  metric_class <- R6::R6Class(
    classname = name,
    inherit = LuzMetric,
    public = public,
    parent_env = parent_env,
    lock_objects = FALSE
  )
  function(...) {
    R6::R6Class(
      inherit = metric_class,
      public = list(
        initialize = function() {
          super$initialize(...)
        }
      ),
      lock_objects = FALSE
    )
  }
}

luz_metric_accuracy <- luz_metric(
  abbrev = "Acc",
  initialize = function() {
    self$correct <- 0
    self$total <- 0
  },
  update = function(preds, target) {
    pred <- torch::torch_argmax(preds, dim = 2)
    self$correct <- self$correct + (pred == target)$
      to(dtype = torch::torch_float())$
      sum()$
      item()
    self$total <- self$total + pred$numel()
  },
  compute = function() {
    self$correct/self$total
  }
)

luz_metric_binary_accuracy_with_logits <- luz_metric(
  abbrev = "Acc",
  initialize = function() {
    self$correct <- 0
    self$total <- 0
    self$threshold <- 0.5
  },
  update = function(preds, targets) {
    preds <- (torch::torch_sigmoid(preds) > self$threshold)$
      to(dtype = torch::torch_float())
    self$correct <- self$correct + (preds == targets)$
      to(dtype = torch::torch_float())$
      sum()$
      item()
    self$total <- self$total + preds$numel()
  },
  compute = function() {
    self$correct/self$total
  }
)


luz_metric_loss_average <- luz_metric(
  abbrev = "Loss",
  initialize = function() {
    self$values <- list()
  },
  update = function(preds, targets) {
    if (length(ctx$loss) == 1)
      loss <- ctx$loss[[1]]
    else
      loss <- ctx$loss

    self$values[[length(self$values) + 1]] <- loss
  },
  average_metric = function(x) {
    if (is.numeric(x[[1]]) || inherits(x[[1]], "torch_tensor"))
      x <- sapply(x, to_numeric)

    if (is.numeric(x)) {
      mean(x)
    } else if (is.list(x)) {
      lapply(purrr::transpose(x), self$average_metric)
    } else if (is.null(x)) {
      NULL
    } else {
      rlang::abort(c(
        "Average metric requires numeric tensor or values or list of them.")
      )
    }
  },
  compute = function() {
    self$average_metric(self$values)
  }
)

to_numeric <- function(x) {
  if (is.numeric(x))
    x
  else if (inherits(x, "torch_tensor"))
    as.numeric(x$to(device = "cpu"))
  else
    rlang::abort("Expected a numeric value or a tensor.")
}


