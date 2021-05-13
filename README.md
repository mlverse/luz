# luz

<!-- badges: start -->
[![R-CMD-check](https://github.com/mlverse/luz/workflows/R-CMD-check/badge.svg)](https://github.com/mlverse/luz/actions)
[![Codecov test coverage](https://codecov.io/gh/mlverse/luz/branch/master/graph/badge.svg)](https://codecov.io/gh/mlverse/luz?branch=master)
[![Discord](https://img.shields.io/discord/837019024499277855?logo=discord)](https://discord.gg/s3D5cKhBkx)
<!-- badges: end -->

luz is a higher level API for torch providing abstractions to allow for much less verbose training loops.

This package is in very early stage of development. Don't use for anything meaningful yet.

It's heavily inspired in other higher level frameworks for deep learning, to cite a few:

-   [FastAI](https://docs.fast.ai/): we are heavily inspired in the FastAI library, specially the `Learner` object and the callbacks API.

-   [Keras](https://keras.io/): We are also heavily inspired by Keras, specially callback names, the lightning module interface is similar to `compile` too.

-   [PyTorch Lightning](https://www.pytorchlightning.ai/): The idea of the `luz_module` being a subclass of `nn_module` is inspired in the **`LightningModule`** object in lightning.

-   [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/): The internal device placement API is heavily inspired in Accelerate, but much more modest in features. Currenly only CPU and Single GPU are supported.

## Installation

Luz is not yet available on CRAN. You can install the development version with:

```{.r}
remotes::install_github("mlverse/luz")
```

## Example

Luz let's you take your Torch `nn_module` definition and `fit` it to a dataloader, while
handling the boring parts like moving data between devices, updating the weights, 
showing progress bars and tracking metrics.

Here's an example defining and training an Autoencoder for the MNIST dataset.
We selected parts of the code to highlight Luz functionality. You can find the
full example code [here](https://mlverse.github.io/luz/articles/examples/mnist-autoencoder.html).

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

Now that we have defined the Autoencder architecture using `torch::nn_module()`,
we can fit it using Luz:

```{.r}
fitted <- net %>%
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam
  ) %>%
  fit(train_dl, epochs = 1, valid_data = test_dl)
```
