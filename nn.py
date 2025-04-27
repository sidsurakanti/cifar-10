# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.v2 import ToImage, ToDtype

from matplotlib import pyplot as plt
import numpy as np

# %%
print(torch.cuda.is_available(), torch.cuda.get_device_name())

# %%
transforms = transforms.Compose([
    ToImage(),
    ToDtype(torch.float32, scale=True)
])

# %%
trainD = datasets.CIFAR10(root="data", train=True, transform=transforms, download=True)
testD = datasets.CIFAR10(root="data", train=False, transform=transforms, download=True)

# %%
x, label = trainD[np.random.randint(len(trainD))] # 32x32 PIL image, label
plt.imshow(x.permute(1, 2, 0))
print(label, x.size(), type(x))

# %%
BATCH_SIZE = 512
DEVICE = torch.accelerator.current_accelerator()

trainDL = DataLoader(trainD, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
testDL = DataLoader(trainD, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

# %%
class Cifar(nn.Module):
  def __init__(self):
    super().__init__()
    self.labels = {
                    0: "airplane",  
                    1: "automobile", 
                    2: "bird",
                    3: "cat",
                    4: "deer",
                    5: "dog",
                    6: "frog",
                    7: "horse",
                    8: "ship",
                    9: "truck"
                  }
    self.model = nn.Sequential(
      nn.Conv2d(3, 10, 3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),

      nn.Conv2d(10, 20, 5),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),

      nn.Flatten(),
      nn.Linear(720, 256),
      nn.ReLU(),

      nn.Linear(256, 10)
    )

  def forward(self, x):
    logits = self.model(x)
    return logits

# %%
def train(dataloader, model, loss_fn, optimizer):
  model.train()
  
  for batch_i, (X, y) in enumerate(dataloader):
    X, y = X.to(DEVICE), y.to(DEVICE)
    # get predictions
    preds = model(X)
    # calculate loss
    loss = loss_fn(preds, y)
    # backprop
    loss.backward()
    # gradient descent 
    optimizer.step()    
    optimizer.zero_grad()

    if batch_i % 25 < 1:
      print(f"Batch {batch_i}/{len(dataloader)}, Loss: {loss.item():.4f}")

def test(dataloader, model, loss_fn):
  model.eval()
  loss_t = correct = 0
  size, num_batches = len(dataloader.dataset), len(dataloader)
  
  # run through testing data
  with torch.no_grad():
    for batch_i, (X, y) in enumerate(dataloader):
      X, y = X.to(DEVICE), y.to(DEVICE)

      # get model preds
      preds = model(X)
      loss_t += loss_fn(preds, y).item()
      correct += (preds.argmax(dim=1) == y).type(torch.float).sum().item()
    
  # calculate average loss & accuracy
  avg_loss = loss_t / num_batches  
  accuracy = correct / size * 100

  print(f"TEST, Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")

  return accuracy, avg_loss
  

def fit(epochs: int):
  model = Cifar().to(DEVICE)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  loss_fn = torch.nn.CrossEntropyLoss()

  accuracies, losses = [], []

  print("Starting...")
  for epoch in range(epochs):
    print("\nEpoch", epoch+1)

    train(trainDL, model, loss_fn, optimizer)
    acc, loss = test(testDL, model, loss_fn )

    accuracies.append(acc)
    losses.append(loss)

  torch.save(model.state_dict(), "model_weights.pth")
  print("Done!\n Weights saved to 'model_weights.pth'")
  return accuracies, losses


# %%
EPOCHS = 50

acc, loss = fit(EPOCHS)


# %%
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle("Performance")

epochs = range(1, EPOCHS+1)

print(acc)
ax1.plot(epochs, acc)
ax1.set_title("Accuracy")

ax2.plot(epochs, loss)
ax1.set_title("Loss")

# %%



