# luz

<!-- badges: start -->
[![R-CMD-check](https://github.com/mlverse/luz/workflows/R-CMD-check/badge.svg)](https://github.com/mlverse/luz/actions)
[![Codecov test coverage](https://codecov.io/gh/mlverse/luz/branch/main/graph/badge.svg)](https://codecov.io/gh/mlverse/luz?branch=main)
[![Discord](https://img.shields.io/discord/837019024499277855?logo=discord)](https://discord.com/invite/s3D5cKhBkx)
[![CRAN status](https://www.r-pkg.org/badges/version/luz)](https://CRAN.R-project.org/package=luz)
[![](https://cranlogs.r-pkg.org/badges/luz)](https://cran.r-project.org/package=luz)
<!-- badges: end -->

Luz is a higher level API for torch providing abstractions to allow for much less verbose training loops.

This package is still under development.

It is heavily inspired by other higher level frameworks for deep learning, to cite a few:

-   [FastAI](https://docs.fast.ai/): we are heavily inspired by the FastAI library, especially the `Learner` object and the callbacks API.

-   [Keras](https://keras.io/): We are also heavily inspired by Keras, especially callback names. The lightning module interface is similar to `compile`, too.

-   [PyTorch Lightning](https://www.pytorchlightning.ai/): The idea of the `luz_module` being a subclass of `nn_module` is inspired by the **`LightningModule`** object in lightning.

-   [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/): The internal device placement API is heavily inspired by Accelerate, but is much more modest in features. Currently only CPU and Single GPU are supported.

## Installation

You can install the released version from CRAN with:

```{.r}
install.packages("luz")
```

or the development version with:

```{.r}
remotes::install_github("mlverse/luz")
```

## Example

Luz lets you take your torch `nn_module` definition and `fit` it to a dataloader, while
handling the boring parts like moving data between devices, updating the weights, 
showing progress bars and tracking metrics.

Here's an example defining and training an Autoencoder for the MNIST dataset.
We selected parts of the code to highlight luz functionality. 
You can find the full example code [here](https://mlverse.github.io/luz/articles/examples/mnist-autoencoder.html).

```{.r}
net <- nn_module(
  "Net",
  initialize = function() {
    self$encoder <- nn_sequential(
      nn_conv2d(1, 6, kernel_size=5),
      nn_relu(),
      nn_conv2d(6, 16, kernel_size=5),
      nn_relu()
    )
    self$decoder <- nn_sequential(
      nn_conv_transpose2d(16, 6, kernel_size = 5),
      nn_relu(),
      nn_conv_transpose2d(6, 1, kernel_size = 5),
      nn_sigmoid()
    )
  },
  forward = function(x) {
    x %>%
      self$encoder() %>%
      self$decoder()
  }
)
```

Now that we have defined the Autoencoder architecture using `torch::nn_module()`, we can fit it using luz:

```{.r}
fitted <- net %>%
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam
  ) %>%
  fit(train_dl, epochs = 1, valid_data = test_dl)
```
