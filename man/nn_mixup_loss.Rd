% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/losses.R
\name{nn_mixup_loss}
\alias{nn_mixup_loss}
\title{Loss to be used with \code{callbacks_mixup()}.}
\usage{
nn_mixup_loss(loss)
}
\arguments{
\item{loss}{the underlying loss \code{nn_module} to call. It must
support the \code{reduction} field. During training the attribute will be changed to
\code{'none'} so we get the loss for individual observations. See for for example
documentation for the \code{reduction} argument in \code{\link[torch:nn_cross_entropy_loss]{torch::nn_cross_entropy_loss()}}.}
}
\description{
In the training phase, computes individual losses with regard to two targets, weights them item-wise,
and averages the linear combinations to yield the mean batch loss.
For validation and testing, defers to the passed-in loss.
}
\details{
It should be used together with \code{\link[=luz_callback_mixup]{luz_callback_mixup()}}.
}
\seealso{
\code{\link[=luz_callback_mixup]{luz_callback_mixup()}}
}
