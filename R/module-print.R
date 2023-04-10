#' @export
print.luz_module_generator <- function(x, ...) {
  cli::cat_line("<luz_module_generator>")
}

#' @export
print.luz_module_fitted <- function(x, ...) {

  has_valid <- !is.null(x$records$profile$valid_batch)

  cli::cat_line("A `luz_module_fitted`")
  cli::cat_rule("Time")
  cli::cat_bullet(
    "Total time: ", prettyunits::pretty_sec(get_total_time(x))
  )
  cli::cat_bullet(
    "Avg time per training epoch: ", prettyunits::pretty_sec(get_average_time(x, "train"))
  )
  if (has_valid) {
    cli::cat_bullet(
      "Avg time per validation epoch ", prettyunits::pretty_sec(get_average_time(x, "valid"))
    )
  }

  cli::cat_line()
  cli::cat_rule("Results")
  cli::cat_line("Metrics observed in the last epoch.")
  cli::cat_line()
  cli::cat_bullet("Training:", bullet = "info", bullet_col = "blue")
  purrr::iwalk(get_formatted_metrics(x, "train"), function(x, nm) {
    cli::cat_line(nm, ": ", x)
  })

  if (has_valid) {
    cli::cat_bullet("Validation:", bullet = "info", bullet_col = "blue")
    purrr::iwalk(get_formatted_metrics(x, "valid"), function(x, nm) {
      cli::cat_line(nm, ": ", x)
    })
  }

  cli::cat_line()
  cli::cat_rule("Model")
  print(x$model)
}

#' @export
print.luz_module_evaluation <- function(x, ...) {
  cli::cat_line("A `luz_module_evaluation`")
  cli::cat_rule("Results")
  purrr::iwalk(get_formatted_metrics(x, "valid"), function(x, nm) {
    cli::cat_line(nm, ": ", x)
  })
}


get_log <- function(object, what, set, index = NULL) {
  if (is.null(index)) {
    index <- length(object$records[[what]][[set]])
  }

  val <- object$records[[what]][[set]]

  if (length(val) < index)
    return(NULL)

  val[[index]]
}

get_all_metrics <- function(object, set, epoch = NULL) {
  if (is.null(epoch)) {
    epoch <- length(object$records[["metrics"]][[set]])
  }
  get_log(object, "metrics", set, epoch)
}

get_metric <- function(object, name, set, epoch= NULL) {
  get_all_metrics(object, set, epoch)[[name]]
}

get_formatted_metrics <- function(object, set, epoch = NULL) {
  values <- get_all_metrics(object, set, epoch)
  for (i in seq_along(values)) {
    values[[i]] <- object$model$metrics[[set]][[i]]$new()$format(values[[i]])
  }
  values
}


