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
    "Avg time per training batch: ", prettyunits::pretty_sec(get_average_time(x, "train_batch"))
  )
  if (has_valid) {
    cli::cat_bullet(
      "Avg time per validation batch ", prettyunits::pretty_sec(get_average_time(x, "valid_batch"))
    )
  }

  cli::cat_line()
  cli::cat_rule("Results")
  cli::cat_line("Metrics observed in the last epoch.")
  cli::cat_line()
  cli::cat_bullet("Training:", bullet = "info", bullet_col = "blue")
  purrr::iwalk(x$ctx$get_formatted_metrics("train"), function(x, nm) {
    cli::cat_line(nm, ": ", x)
  })

  if (has_valid) {
    cli::cat_bullet("Validation:", bullet = "info", bullet_col = "blue")
    purrr::iwalk(x$ctx$get_formatted_metrics("valid"), function(x, nm) {
      cli::cat_line(nm, ": ", x)
    })
  }

  cli::cat_line()
  cli::cat_rule("Model")
  print(x$model)
}



