# PyTorch

**Introduction to Neural Networks with PyTorch summary**

From “what is a neural network?” to “how do I train one in PyTorch?” and then to “how do I use it for image classification?”

Key takeaways:

- A neural network learns by adjusting **weights**
- Inputs go forward through layers to make a prediction
- Error/loss tells the model how wrong it is
- Backpropagation computes how each weight contributed to the error
- Gradient descent updates the weights to reduce the error
- In PyTorch, a lot of this is automated, but I still need to understand the flow

---

**1. Introduction to Neural Networks**

- A neural network is a stack of layers that transforms inputs into outputs
- Each neuron does roughly:

```python
output= activation(weighted_sum+ bias)
```

- Common ideas:
    - weights
    - bias
    - activation functions
    - hidden layers
    - output layer
- Why activations matter:
    - without them, the network is basically just a linear model
- Main activation you saw a lot:

```python
sigmoid
```

and later also:

```python
ReLUsoftmax
```

What matters for the project:

- input layer = image features
- hidden layers = learn useful patterns
- output layer = class scores/probabilities

---

**2. Implementing Gradient Descent**

- This is the “how learning actually happens” lesson
- The network:
    1. makes a prediction
    2. compares prediction to true answer
    3. computes loss/error
    4. backpropagates error
    5. updates weights

Key ideas:

- **forward pass** = compute prediction
- **loss** = how wrong prediction is
- **backward pass** = compute gradients
- **gradient descent** = update weights in the direction that reduces loss

Core update idea:

```python
weight= weight- learning_rate* gradient
```

What to remember:

- learning rate too big → unstable training
- learning rate too small → training is very slow

What matters for the project:

- even if PyTorch does backprop for you, this is the logic behind:

```python
loss.backward()optimizer.step()
```

---

**3. Training Neural Networks**

- This lesson is about training in a more realistic way
- You split data into:
    - training set
    - validation set
    - test set

Why:

- training set = learn parameters
- validation set = tune/check during training
- test set = final unbiased evaluation

Important concepts:

- **overfitting** = model memorizes training data, performs worse on new data
- **underfitting** = model is too simple or not trained enough
- **dropout** = randomly turn off some neurons during training to reduce overfitting
- **epochs** = full passes through the training data

What matters for the project:

- track both training loss and validation accuracy/loss
- don’t judge the model only by training performance
- use evaluation mode when validating/testing:

```python
model.eval()
```

and switch back for training:

```python
model.train()
```

---

**4. Deep Learning with PyTorch** This is the most practical lesson for your project.

Main PyTorch pieces:

- `torch` = tensors and core operations
- `nn` = neural network layers and loss functions
- `optim` = optimizers like Adam or SGD
- `torchvision` = image datasets, transforms, pretrained models

Typical training flow:

```python
model=...criterion=...optimizer=...
for images, labelsin trainloader:    optimizer.zero_grad()    outputs= model(images)    loss= criterion(outputs, labels)    loss.backward()    optimizer.step()
```

Important PyTorch ideas:

- tensors are like NumPy arrays with GPU support
- autograd tracks operations so gradients can be computed automatically
- `nn.Module` is the base class for models
- `nn.Sequential` is a quick way to stack layers
- `state_dict()` stores model weights

What matters most for the project:

- image transforms
- dataloaders
- defining a classifier
- training loop
- validation loop
- saving/loading model

---



**Examples and patterns to keep for practical use / programming neural networks with pytorch**

**1. Image transforms** Use transforms to convert images into tensors and normalize them. Training data often also gets augmentation.

Example:

```python
from torchvisionimport transforms
train_transforms= transforms.Compose([    transforms.RandomRotation(30),    transforms.RandomResizedCrop(224),    transforms.RandomHorizontalFlip(),    transforms.ToTensor(),    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
test_transforms= transforms.Compose([    transforms.Resize(256),    transforms.CenterCrop(224),    transforms.ToTensor(),    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
```

Why:

- augmentation helps generalization
- normalization matches what pretrained models expect

Rubric-relevant:

- use transforms
- include at least one augmentation for training

---

**2. Load datasets and dataloaders** Example:

```python
from torchvisionimport datasetsfrom torch.utils.dataimport DataLoader
train_data= datasets.ImageFolder(train_dir, transform=train_transforms)test_data= datasets.ImageFolder(test_dir, transform=test_transforms)
trainloader= DataLoader(train_data, batch_size=64, shuffle=True)testloader= DataLoader(test_data, batch_size=64, shuffle=False)
```

If you also use validation:

```python
valid_data= datasets.ImageFolder(valid_dir, transform=test_transforms)validloader= DataLoader(valid_data, batch_size=64, shuffle=False)
```

Why:

- `ImageFolder` reads class folders automatically
- `DataLoader` gives batches for training/testing

Rubric-relevant:

- use `DataLoader` for train and test
- validation loader is a good extra

---

**3. Inspect the dataset** Useful checks:

```python
print(len(train_data))print(len(test_data))print(train_data.classes)print(train_data.class_to_idx)
```

Show one image:

```python
import matplotlib.pyplotas plt
images, labels=next(iter(testloader))plt.imshow(images[0].permute(1,2,0))plt.show()
```

If normalized, you usually need to unnormalize before displaying nicely.

Rubric-relevant:

- show dataset size/shape
- display at least one test image

---

**4. Build a model** Simple feedforward example:

```python
import torch.nnas nn
model= nn.Sequential(    nn.Flatten(),    nn.Linear(3*224*224,512),    nn.ReLU(),    nn.Dropout(0.2),    nn.Linear(512,10))
```

If using log probabilities:

```python
model= nn.Sequential(    nn.Flatten(),    nn.Linear(3*224*224,512),    nn.ReLU(),    nn.Dropout(0.2),    nn.Linear(512,10),    nn.LogSoftmax(dim=1))
```

What to remember:

- final output size must match number of classes
- for 10 classes, output layer needs 10 outputs

Rubric-relevant:

- model outputs class probabilities/scores for all classes

---

**5. Loss function and optimizer** Examples:

```python
import torch.optimas optimimport torch.nnas nn
criterion= nn.CrossEntropyLoss()optimizer= optim.Adam(model.parameters(), lr=0.003)
```

If using `LogSoftmax` in the model:

```python
criterion= nn.NLLLoss()
```

What to remember:

- `CrossEntropyLoss` usually expects raw logits
- `NLLLoss` pairs with `LogSoftmax`

Rubric-relevant:

- use a classification loss
- use an optimizer from `torch.optim`

---

**6. Basic training loop** Reference pattern:

```python
epochs=5train_losses=[]
for epochinrange(epochs):    running_loss=0    model.train()
for images, labelsin trainloader:        optimizer.zero_grad()        outputs= model(images)        loss= criterion(outputs, labels)        loss.backward()        optimizer.step()
        running_loss+= loss.item()
    epoch_loss= running_loss/len(trainloader)    train_losses.append(epoch_loss)print(f"Epoch{epoch+1}/{epochs} - Loss:{epoch_loss:.4f}")
```

What to remember:

- zero gradients first
- forward pass
- compute loss
- backward pass
- optimizer step

Rubric-relevant:

- use training dataloader
- compute batch loss
- record average loss per epoch

---

**7. Validation / evaluation loop** Reference pattern:

```python
model.eval()correct=0total=0test_loss=0
with torch.no_grad():for images, labelsin testloader:        outputs= model(images)        loss= criterion(outputs, labels)        test_loss+= loss.item()
        _, predicted= outputs.max(1)        total+= labels.size(0)        correct+=(predicted== labels).sum().item()
accuracy= correct/ totalavg_test_loss= test_loss/len(testloader)
print("Test Loss:", avg_test_loss)print("Test Accuracy:", accuracy)
```

Why:

- `model.eval()` turns off dropout behavior
- `torch.no_grad()` saves memory and avoids gradient tracking

Rubric-relevant:

- use test dataloader
- compare predictions with true labels

---

**8. Plot loss** Example:

```python
import matplotlib.pyplotas plt
plt.plot(train_losses)plt.xlabel("Epoch")plt.ylabel("Average Loss")plt.title("Training Loss")plt.show()
```

Rubric-relevant:

- plot average loss across epochs

---

**9. Save the trained model** Simple version:

```python
import torch
torch.save(model.state_dict(),"model_weights.pth")
```

Load later:

```python
model.load_state_dict(torch.load("model_weights.pth"))model.eval()
```

Rubric-relevant:

- save trained weights with `torch.save()`

---

**10. If your project version uses checkpoints** A more complete save pattern:

```python
checkpoint={'model_state_dict': model.state_dict(),'class_to_idx': train_data.class_to_idx}
torch.save(checkpoint,'checkpoint.pth')
```

Load pattern:

```python
checkpoint= torch.load('checkpoint.pth')model.load_state_dict(checkpoint['model_state_dict'])class_to_idx= checkpoint['class_to_idx']model.eval()
```

Why:

- weights alone are not always enough
- class mapping is often needed for prediction output

---

**11. If your project version uses pretrained models / transfer learning** Common pattern:

```python
from torchvisionimport models
model= models.resnet18(pretrained=True)
for paramin model.parameters():    param.requires_grad=False
```

Replace classifier:

```python
model.fc= nn.Linear(model.fc.in_features,10)
```

Optimizer only for classifier:

```python
optimizer= optim.Adam(model.fc.parameters(), lr=0.003)
```

What to remember:

- freeze pretrained feature extractor
- replace final classifier layer
- train only the new classifier first

This is often the most practical route for image classification.

---

**12. Prediction pattern** If you later need top-k predictions:

```python
with torch.no_grad():    outputs= model(image_tensor)    probs= torch.softmax(outputs, dim=1)    top_probs, top_classes= probs.topk(5, dim=1)
```

What it gives:

- top predicted probabilities
- top predicted class indices

---

**13. Common mistakes to avoid**

- forgetting `model.train()` during training
- forgetting `model.eval()` during validation/testing
- using the wrong loss for the model output
- mismatch between output layer size and number of classes
- not normalizing images
- not saving class mapping when needed
- plotting total loss instead of average loss
- evaluating only on training data

---

**Project-focused memory summary** For this project, I mainly need to remember:

- transform images
- create datasets and dataloaders
- build a classifier
- choose classification loss + optimizer
- train with forward → loss → backward → step
- evaluate with `model.eval()` and `torch.no_grad()`
- compute accuracy
- plot average loss
- save model weights/checkpoint

---

**Very short version**

- Neural nets learn weights from data
- Backprop + gradient descent are how learning happens
- PyTorch automates gradients and optimization
- For the project: preprocess images, batch them with dataloaders, train a classifier, evaluate accuracy, save the model