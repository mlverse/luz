% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/metrics.R
\name{luz_metric_binary_accuracy}
\alias{luz_metric_binary_accuracy}
\title{Binary accuracy}
\usage{
luz_metric_binary_accuracy(threshold = 0.5)
}
\arguments{
\item{threshold}{value used to classifiy observations between 0 and 1.}
}
\value{
Returns new luz metric.
}
\description{
Computes the accuracy for binary classification problems where the
model returns probabilities. Commonly used when the loss is \code{\link[torch:nn_bce_loss]{torch::nn_bce_loss()}}.
}
\examples{
if (torch::torch_is_installed()) {
library(torch)
metric <- luz_metric_binary_accuracy(threshold = 0.5)
metric <- metric$new()
metric$update(torch_rand(100), torch::torch_randint(0, 1, size = 100))
metric$compute()
}

}
\seealso{
Other luz_metrics: 
\code{\link{luz_metric}()},
\code{\link{luz_metric_accuracy}()},
\code{\link{luz_metric_binary_accuracy_with_logits}()},
\code{\link{luz_metric_binary_auroc}()},
\code{\link{luz_metric_mae}()},
\code{\link{luz_metric_mse}()},
\code{\link{luz_metric_multiclass_auroc}()},
\code{\link{luz_metric_rmse}()}
}
\concept{luz_metrics}
