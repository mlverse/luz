# luz

<!-- badges: start -->
[![R-CMD-check](https://github.com/mlverse/luz/workflows/R-CMD-check/badge.svg)](https://github.com/mlverse/luz/actions)
[![Codecov test coverage](https://codecov.io/gh/mlverse/luz/branch/master/graph/badge.svg)](https://codecov.io/gh/mlverse/luz?branch=master)
<!-- badges: end -->

luz is a higher level API for torch providing abstractions to allow for much less verbose training loops.

This package is in very early stage of development. Don't use for anything meaningful yet.

It's heavily inspired in other higher level frameworks for deep learning, to cite a few:

-   [FastAI](https://docs.fast.ai/): we are heavily inspired in the FastAI library, specially the `Learner` object and the callbacks API.

-   [Keras](https://keras.io/): We are also heavily inspired by Keras, specially callback names, the lightning module interface is similar to `compile` too.

-   [PyTorch Luzning](https://www.pyluzning.ai/): The idea of the `luz_module` being a subclass of `nn_module` is inspired in the **`LuzningModule`** object in lightning.

-   [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/): The internal device placement API is heavily inspired in Accelerate, but much more modest in features. Currenly only CPU and Single GPU are supported.

## Todo

-   [ ] 'compiling' and training classification models

-   [ ] training and validation data

-   [ ] metrics other than the loss

-   [ ] callbacks for logging and progressbar

-   [ ] custom optimizer definition

-   [ ] custom training and validation steps

-   [ ] timings for each part of the model

-   [ ] handle device placement

## Installation

You can install the released version of luz from [CRAN](https://CRAN.R-project.org) with:

``` {.r}
install.packages("luz")
```

## Example

This is a basic example which shows you how to solve a common problem:

``` {.r}
library(luz)
## basic example code
```
