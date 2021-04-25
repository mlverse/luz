LightMetric <- R6::R6Class(
  "LightMetric",
  lock_objects = FALSE,
  public = list(
    format = function(v) {
      round(v, 4)
    }
  )
)

light_metric <- function(name = NULL, ..., public, active, parent_env = parent.frame()) {
  public <- rlang::list2(...)
  R6::R6Class(
    classname = name,
    inherit = LightMetric,
    public = public,
    parent_env = parent_env,
    lock_objects = FALSE
  )
}

light_metric_accuracy <- light_metric(
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


