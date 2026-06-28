import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset
import torch
import torchvision
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
import h5py
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset



class STEMNISTDataset(Dataset):
    def __init__(self, root_dir, T=40, H=16, W=16, transform=None):
        self.transform = transform
        self.samples = []
        self.T = T
        self.H = H
        self.W = W

        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

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

        frames = spikes_to_frames(spikes, path, T=self.T, H=self.H, W=self.W)
        frames = frames / (frames.max() + 1e-8)
        frames = torch.tensor(frames, dtype=torch.float32)
        return frames, label


def spikes_to_frames(spikes, path, T=40, H=16, W=16):
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

def forward_pass(net, data):
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(data.size(1)):  # T
      spk_out, mem_out = net(data[:, step])
      spk_rec.append(spk_out)

  return torch.stack(spk_rec)

# sensor_size = tonic.datasets.NMNIST.sensor_size

# Denoise removes isolated, one-off events
# time_window
# frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
#                                       transforms.ToFrame(sensor_size=(16,16),
#                                                          time_window=1000)
#                                      ])

batch_size = 128
data_path=r"C:\Users\zhanr\OneDrive\Desktop\SNN\STEMNIST Dataset\ProcessedSpikes"

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

dataset = STEMNISTDataset(data_path)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# train_dataset, test_dataset = random_split(dataset,[train_size, test_size],generator=torch.Generator().manual_seed(42))
labels = [label for _, label in dataset.samples]
indices = list(range(len(dataset)))
train_idx, test_idx = train_test_split(indices,test_size=0.2,random_state=42,stratify=labels)
train_dataset = Subset(dataset, train_idx)
test_dataset = Subset(dataset, test_idx)

# train_dataset = STEMNISTDataset(data_path)
# test_dataset = STEMNISTDataset(data_path)
# test_dataset = datasets.MNIST('/tmp/data/mnist', train=False, download=True, transform=transform)

# trainset = STEMNISTDataset(root_dir=r"C:\Users\zhanr\OneDrive\Desktop\SNN\STEMNIST Dataset\ProcessedSpikes",T=20, H=16, W=16)
# testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)

# transform = tonic.transforms.Compose([torch.from_numpy, torchvision.transforms.RandomRotation([-10,10])])

# train_dataset.transform = transform
# cached_trainset = DiskCachedDataset(train_dataset, cache_path=r"C:\Users\zhanr\OneDrive\Desktop\SNN\cache\train")

# no augmentations for the testset
# cached_testset = DiskCachedDataset(test_dataset, cache_path=r"C:\Users\zhanr\OneDrive\Desktop\SNN\cache\test")

# trainloader = DataLoader(cached_trainset, batch_size=128, shuffle=True)
# testloader = DataLoader(cached_testset, batch_size=128, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)



event_tensor, target = next(iter(train_loader))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# neuron and simulation parameters
spike_grad = surrogate.atan()
beta = 0.9

#  Initialize Network
net = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(64*4*4, 256),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Linear(256, 35),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
num_epochs = 50

loss_hist = []
acc_hist = []

# training loop
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        net.train()
        angle = torch.empty(1).uniform_(-10, 10).item()
        for t in range(data.size(1)):
            frame = data[:, t]
            # angle = torch.empty(1).uniform_(-10, 10).item()
            data[:, t] = torchvision.transforms.functional.rotate(frame, angle)
        spk_rec = forward_pass(net, data)
        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

        acc = SF.accuracy_rate(spk_rec, targets)
        acc_hist.append(acc)
        print(f"Accuracy: {acc * 100:.2f}%\n")

    net.eval()

    test_acc = 0
    test_loss = 0
    num_batches = 0

    with torch.no_grad():

        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)

            spk_rec = forward_pass(net, data)

            loss = loss_fn(spk_rec, targets)

            test_loss += loss.item()
            test_acc += SF.accuracy_rate(spk_rec, targets)

            num_batches += 1

    test_acc /= num_batches
    test_loss /= num_batches

    print(
        f"Epoch {epoch}: "
        f"Test Loss={test_loss:.4f}, "
        f"Test Acc={test_acc * 100:.2f}%"
    )

# Plot Loss
fig = plt.figure(facecolor="w")
plt.plot(acc_hist)
plt.title("Train Set Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()

spk_rec = forward_pass(net, data)