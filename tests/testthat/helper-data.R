# Helper train dataloaders and models for tests

get_ds <- torch::dataset(
  initialize = function(len = 100, x_size = 10, y_size = 1, fixed_values = FALSE) {
    self$len <- len
    self$x <- torch::torch_randn(size = c(len, x_size))
    self$y <- torch::torch_randn(size = c(len, y_size))
  },
  .getitem = function(i) {
    list(
      x = self$x[i,..],
      y = self$y[i,..]
    )
  },
  .length = function() {
    self$len
  }
)

get_binary_ds <- torch::dataset(
  inherit = get_ds,
  initialize = function(...) {
    super$initialize(...)
    self$y <- torch::torch_randint(low = 0,high = 2, size = self$y$shape)
  }
)

get_dl <- function(batch_size = 10, ...) {
  torch::dataloader(get_ds(...), batch_size = batch_size)
}

get_binary_dl <- function(batch_size = 10,...) {
  torch::dataloader(get_binary_ds(...), batch_size = batch_size)
}

get_test_dl <- function(batch_size = 10, ...) {
  torch::dataloader(get_ds(...), batch_size = batch_size, shuffle = FALSE)
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



