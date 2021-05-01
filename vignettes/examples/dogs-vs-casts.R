# Packages ----------------------------------------------------------------
library(torch)
library(torchvision)
library(torchdatasets)

set.seed(1)

# Datasets and loaders ----------------------------------------------------

dir <- "~/Downloads/dogs-vs-cats" #caching directory

ds <- torchdatasets::dogs_vs_cats_dataset(
  dir,
  token = "~/Downloads/kaggle.json",
  transform = . %>%
    transform_to_tensor() %>%
    transform_resize(size = c(224, 224)),
  target_transform = function(x) as.double(x) - 1
)

train_id <- sample.int(length(ds), size = 0.7*length(ds))
train_ds <- dataset_subset(ds, indices = train_id)
valid_ds <- dataset_subset(ds, indices = which(!seq_along(ds) %in% train_id))

train_dl <- dataloader(train_ds, batch_size = 128, shuffle = TRUE)
valid_dl <- dataloader(valid_ds, batch_size = 128)

# Buildifng the network ---------------------------------------------------

net <- nn_module(
  inherit = torchvision:::alexnet,
  forward = function(x) {
    super$forward(x)[,1]
  }
)

# Train -------------------------------------------------------------------

fitted <- net %>%
  setup(
    loss = nn_bce_with_logits_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz:::luz_metric_binary_accuracy_with_logits
    )
  ) %>%
  set_hparams(num_classes = 1) %>%
  luz:::fit.luz_module_generator(train_dl, epochs = 10, valid_data = valid_dl)
