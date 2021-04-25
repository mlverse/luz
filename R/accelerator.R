LightAccelerator <- R6::R6Class(
  classname = "LightAccelerator",
  lock_objects = FALSE,
  public = list(
    initialize = function(device_placement = TRUE, cpu = FALSE) {
      self$device_placement = device_placement
      self$state <- LightAcceleratorState$new(cpu = cpu)
    },
    prepare = function(...) {

      objs <- rlang::list2(...)
      old_parameter_ids <- names(get_parameter_ids(!!objs))

      results <- lapply(objs, self$prepare_one)
      new_parameter_ids <- get_parameter_ids(!!!results)

      mapping <- setNames(new_parameter_ids, old_parameter_ids)

      if (length(old_parameter_ids) != length(new_parameter_ids))
        rlang::abort(c("Wrong number of parameters in the prepared model.",
                       "Please report an issue in the GitHub repository."))

      switch_parameters(!!!results, .mapping = mapping)

    },
    prepare_one = function(obj) {

      if (torch::is_nn_module(obj))
        return(self$prepare_model(obj))

      if (torch::is_optimizer(obj))
        return(self$prepare_optimizer(obj))

    },
    prepare_model = function(model) {
      if (self$device_placement) {
        model <- model$to(device = self$device)
      }
      model
    },
    prepare_optimizer = function(optmizer) {
      # currently we have nothing to do here
    }
  ),
  active = list(
    device = function() {
      self$state$device
    }
  )
)

LightAcceleratorState <- R6::R6Class(
  classname = "LightAcceleratorState",
  lock_objects = FALSE,
  public = list(
    initialize = function(cpu = FALSE) {
      self$device <- if (torch::cuda_is_available() && !cpu) "cuda" else "cpu"
    }
  )
)

get_parameter_ids <- function(..., with_parameters) {
  objs <- rlang::list2(...)
  parameters <- list()

  for (obj in objs) {
    if (torch::is_nn_module(obj)) {
      parameters <- append(parameters, obj$parameters)
    }
  }

  names(parameters) <- sapply(parameters, get_param_id)
  parameters
}

switch_parameters <- function(..., .mapping) {
  objs <- rlang::list2(...)
  for (obj in objs) {
    if (torch::is_optimizer(obj)) {
      obj$param_groups <- lapply(
        obj$param_groups,
        function(x) {
          .mapping[[get_param_id(x)]]
        })
    }
  }
  invisible(NULL)
}

get_param_id <- function(p) {
  p$storage()$data_ptr()
}
