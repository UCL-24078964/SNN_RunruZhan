# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import h5py
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
from torch.utils.data import Dataset, random_split

# Leaky neuron model, overriding the backward pass with a custom function
class LeakySurrogate(nn.Module):
  def __init__(self, beta, threshold=1.0):
      super(LeakySurrogate, self).__init__()

      # initialize decay rate beta and threshold
      self.beta = beta
      self.threshold = threshold
      self.spike_gradient = self.ATan.apply

  # the forward function is called each time we call Leaky
  def forward(self, input_, mem):
    spk = self.spike_gradient((mem-self.threshold))  # call the Heaviside function
    reset = (self.beta * spk * self.threshold).detach()  # remove reset from computational graph
    mem = self.beta * mem + input_ - reset  # Eq (1)
    return spk, mem

  # Forward pass: Heaviside function
  # Backward pass: Override Dirac Delta with the derivative of the ArcTan function
  @staticmethod
  class ATan(torch.autograd.Function):
      @staticmethod
      def forward(ctx, mem):
          spk = (mem > 0).float() # Heaviside on the forward pass: Eq(2)
          ctx.save_for_backward(mem)  # store the membrane for use in the backward pass
          return spk

      @staticmethod
      def backward(ctx, grad_output):
          (mem,) = ctx.saved_tensors  # retrieve the membrane potential
          grad = 1 / (1 + (np.pi * mem).pow_(2)) * grad_output # Eqn 5
          return grad

def spikes_to_frames(spikes, path, T=20, H=16, W=16):
    frames = np.zeros(
        (T, 2, H, W),
        dtype=np.float32
    )

    t0, t1 = spikes["timestamp"].min(), spikes["timestamp"].max()
    bins = np.linspace(t0, t1, T+1)

    for i in range(len(spikes)):
        t = spikes[i]["timestamp"]
        p = spikes[i]["polarity"]
        ti = np.clip(np.searchsorted(bins, t) - 1, 0, T-1)
        if 0 <= ti < T:
            taxel = int(spikes[i]["taxel_id"]) - 1
            x = taxel % W
            y = taxel // H
            if p > 0:
                frames[ti, 0, y, x] += 1
            else:
                frames[ti, 1, y, x] += 1

    return frames

class STEMNISTDataset(Dataset):
    def __init__(self, root_dir, T=20, H=16, W=16, transform=None):
        self.transform = transform
        self.samples = []
        self.T = T
        self.H = H
        self.W = W

        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for c in self.classes:

            class_path = os.path.join(root_dir, c)

            for file in os.listdir(class_path):

                if file.endswith("_spikes.h5"):
                    full_path = os.path.join(class_path, file)

                    self.samples.append(
                        (full_path, self.class_to_idx[c])
                    )

        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]
        with h5py.File(path, "r") as f:
            spikes = f["spikes"][:]

        frames = spikes_to_frames(spikes, path, T=20, H=16, W=16)
        frames = torch.tensor(frames, dtype=torch.float32)
        return frames, label

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden_1)
        self.lif1 = snn.Leaky(beta=beta)
        # self.fc2 = nn.Linear(num_hidden_1, num_hidden_2)
        # self.lif2 = snn.Leaky(beta=beta)
        # self.fc3 = nn.Linear(num_hidden_2, num_outputs)
        # self.lif3 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden_1, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)


    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []
        T = x.size(1)

        for step in range(T):
            x_t = x[:, step]
            x_t = x_t.reshape(x.size(0), -1)
            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec), torch.stack(mem2_rec)

# pass data into the network, sum the spikes over time
# and compare the neuron with the highest number of spikes
# with the target

def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data)
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer():
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")

def evaluate_accuracy(loader):
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():

        for data, targets in loader:

            data = data.to(device)
            targets = targets.to(device)

            spk_rec, _ = net(data)

            _, predicted = spk_rec.sum(dim=0).max(1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    net.train()
    return 100 * correct / total

lif1 = LeakySurrogate(beta=0.9)

# dataloader arguments
batch_size = 128
data_path=r"C:\Users\zhanr\OneDrive\Desktop\SNN\STEMNIST Dataset\ProcessedSpikes"

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Define a transform
# transform = transforms.Compose([
#             transforms.Resize((28, 28)),
#             transforms.Grayscale(),
#             transforms.ToTensor(),
#             transforms.Normalize((0,), (1,))])

dataset = STEMNISTDataset(data_path)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset,[train_size, test_size],generator=torch.Generator().manual_seed(42))

# train_dataset = STEMNISTDataset(data_path)
# test_dataset = STEMNISTDataset(data_path)
# test_dataset = datasets.MNIST('/tmp/data/mnist', train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Network Architecture
num_inputs = 512
num_hidden_1 = 64
# num_hidden_1 = 512
# num_hidden_2 = 128
num_outputs = 35

# Temporal Dynamics
num_steps = 20
beta = 0.95


# Load the network onto CUDA if available
net = Net().to(device)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))

data, targets = next(iter(train_loader))
data = data.to(device)
targets = targets.to(device)

spk_rec, mem_rec = net(data)

# print(mem_rec.size())

# initialize the total loss value
loss_val = torch.zeros((1), dtype=dtype, device=device)

# sum loss at every step
for step in range(num_steps):
  loss_val += loss(mem_rec[step], targets)

# print(f"Training loss: {loss_val.item():.3f}")
# print_batch_accuracy(data, targets, train=True)

# clear previously stored gradients
optimizer.zero_grad()

# calculate the gradients
loss_val.backward()

# weight update
optimizer.step()

# calculate new network outputs using the same data
spk_rec, mem_rec = net(data)

# initialize the total loss value
loss_val = torch.zeros((1), dtype=dtype, device=device)

# sum loss at every step
for step in range(num_steps):
  loss_val += loss(mem_rec[step], targets)

print(f"Training loss: {loss_val.item():.3f}")
print_batch_accuracy(data, targets, train=True)

num_epochs = 50
loss_hist = []
test_loss_hist = []
train_acc_hist = []
test_acc_hist = []
train_loss_hist = []

counter = 0

# Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    # Minibatch training loop
    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        net.train()
        spk_rec, mem_rec = net(data)

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_spk, test_mem = net(test_data)

            # Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                test_loss += loss(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())

            # Print train/test loss/accuracy
            # if counter % 24 == 0:
            #     train_printer()
            counter += 1
            iter_counter +=1
    train_acc = evaluate_accuracy(train_loader)
    test_acc = evaluate_accuracy(test_loader)

    train_acc_hist.append(train_acc)
    test_acc_hist.append(test_acc)
    print(
        f"Epoch {epoch + 1}: "
        f"Train={train_acc:.2f}% "
        f"Test={test_acc:.2f}%"
    )

# Plot Loss
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

epochs = range(1, len(train_acc_hist)+1)
plt.figure(figsize=(8,5))
plt.plot(epochs, train_acc_hist, label="Train Accuracy")
plt.plot(epochs, test_acc_hist, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Train vs Test Accuracy")
plt.legend()
plt.grid()
plt.show()

total = 0
correct = 0

# drop_last switched to False to keep all samples
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

with torch.no_grad():
  net.eval()
  total_loss = 0
  for data, targets in test_loader:
    data = data.to(device)
    targets = targets.to(device)

    # forward pass
    test_spk, test_mem = net(data)

    batch_loss = 0
    # calculate total accuracy
    for step in range(num_steps):
        batch_loss += loss(
            test_mem[step],
            targets
        )

    total_loss += batch_loss.item()
