# Image Classifier Project Rubric

## Data Preparation

### Transform data for use in neural networks.

- Torchvision transforms are specified in a list for use in the data loader.
- At a minimum, `.toTensor()` is included to use the DataLoader with the neural network.

### Augment training data to improve robustness of neural networks.

- At least one data augmentation technique (rotation, cropping, etc.) is implemented as part of the list of transforms for the DataLoader.

### Use DataLoader to feed training data to PyTorch models.

- A DataLoader object for both train and test sets has been created using the train and test sets loaded from torchvision.

### Explore datasets and describe their properties to set and optimize neural network parameters.

- Notebook contains code that shows the size and shape of the training data.
- At least one of the images in the test set is printed to the screen using `plt.imshow()`.

---

## Model Design and Training

### Use PyTorch to build a neural network for image classification.

- A Model class is created and implements a forward method that outputs a prediction probability for each of the 10 classes using softmax.

### Select a loss function for training a classification network.

- A loss function that works for classification tasks is specified.

### Define an optimizer to minimize loss function and update model parameters for improved accuracy.

- An optimizer from `torch.optim` is used to minimize the loss function.

### Train a neural network using PyTorch to achieve a given level of accuracy.

- The training DataLoader is used to feed data to the neural network and compute the batch loss.
- The trained model achieves at least 45% accuracy.

### Compute and plot average loss to track model performance.

- The average value of the loss function at each epoch is recorded and plotted after training is complete.

---

## Model Testing and Evaluation

### Use DataLoaders to test the accuracy of a model.

- The test DataLoader is used to get predictions from the neural network and compare the predictions to the true labels.

### Make decisions based on model results.

- Students explain in no fewer than three sentences how their model's accuracy compares to the hypothetical model proposed and make a recommendation on whether to build or buy an object detection solution.

### Save trained model parameters for later use.

- The `torch.save()` function is used to save the weights of the trained model.
