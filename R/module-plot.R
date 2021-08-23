#' @export
plot.luz_module_fitted <- function(x, ...) {
  check_installed("ggplot2")
  p <- ggplot2::ggplot(x$ctx$get_metrics_df(), ggplot2::aes(x = epoch, y = value))
  p <- p + ggplot2::geom_point() + ggplot2::geom_line()
  p + ggplot2::facet_grid(metric ~ set, scales = "free_y")
}

globalVariables(c("epoch", "value"))
