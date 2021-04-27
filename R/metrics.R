LuzMetric <- R6::R6Class(
  "LuzMetric",
  lock_objects = FALSE,
  public = list(
    format = function(v) {
      round(v, 4)
    }
  )
)

luz_metric <- function(name = NULL, ..., public, active, parent_env = parent.frame()) {
  public <- rlang::list2(...)
  R6::R6Class(
    classname = name,
    inherit = LuzMetric,
    public = public,
    parent_env = parent_env,
    lock_objects = FALSE
  )
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


