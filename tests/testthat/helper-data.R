# Helper train dataloaders and models for tests

get_ds <- torch::dataset(
  initialize = function(len = 100, x_size = 10, y_size = 1) {
    self$len <- len
    self$x_size <- x_size
    self$y_size <- y_size
  },
  .getitem = function(i) {
    list(
      x = torch::torch_randn(size = self$x_size),
      y = torch::torch_randn(size = self$y_size)
    )
  },
  .length = function() {
    self$len
  }
)

get_dl <- function(batch_size = 10, ...) {
  torch::dataloader(get_ds(...), batch_size = batch_size)
}

get_model <- function() {
  torch::nn_module(
    initialize = function(input_size, output_size) {
      self$fc <- torch::nn_linear(prod(input_size), prod(output_size))
      self$output_size <- output_size
    },
    forward = function(x) {
      out <- x %>%
        torch::torch_flatten(start_dim = 2) %>%
        self$fc()
      out$view(c(x$shape[1], self$output_size))
    }
  )
}




