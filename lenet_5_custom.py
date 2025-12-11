# -*- coding: utf-8 -*-

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 1, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.pool = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx*len(data):5d}/{len(train_loader.dataset)} '
                  f'({100.*batch_idx/len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(f'\nAccuracy: {correct}/{len(test_loader.dataset)} ({100.*correct/len(test_loader.dataset):.2f}%)\n')

def main():
    parser = argparse.ArgumentParser(description='LeNet-5 MNIST Lab4')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log-interval', type=int, default=10)   
    parser.add_argument('--no-cuda', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=1000, shuffle=False)

    model = LeNet5().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    #Save model state_dict
    os.makedirs("data", exist_ok=True)
    torch.save(model.state_dict(), "data/lenet5_model.pt")

    # Export 10 files of weigths
    print("\nexporting 10 weight files to 'weights' folder")
    os.makedirs("weights", exist_ok=True)

    np.savetxt("weights/1. convolution 1 (w_conv1.txt)",     model.conv1.weight.data.cpu().numpy().reshape(6, -1),  fmt='%.6f') # reshape để lưu dưới dạng ma trận 2D
    np.savetxt("weights/2. bias của convolution 1 (b_conv1.txt)", model.conv1.bias.data.cpu().numpy(),      fmt='%.6f')
    np.savetxt("weights/3. convolution  convolution 2 (w_conv2.txt)",     model.conv2.weight.data.cpu().numpy().reshape(16, -1), fmt='%.6f')
    np.savetxt("weights/4. bias của convolution 2 (b_conv2.txt)", model.conv2.bias.data.cpu().numpy(),      fmt='%.6f')
    np.savetxt("weights/5. fully connection 1 (w_fc1.txt)",  model.fc1.weight.data.cpu().numpy(),               fmt='%.6f')
    np.savetxt("weights/6. bias của fully connection 1 (b_fc1.txt)", model.fc1.bias.data.cpu().numpy(),    fmt='%.6f')
    np.savetxt("weights/7. fully connection 2 (w_fc2.txt)",  model.fc2.weight.data.cpu().numpy(),               fmt='%.6f')
    np.savetxt("weights/8. bias của fully connection 2 (b_fc2.txt)", model.fc2.bias.data.cpu().numpy(),    fmt='%.6f')
    np.savetxt("weights/9. fully connection 3 (w_fc3.txt)",  model.fc3.weight.data.cpu().numpy(),               fmt='%.6f')
    np.savetxt("weights/10. bias của fully connection 3 (b_fc3.txt)", model.fc3.bias.data.cpu().numpy(),   fmt='%.6f')

    print("Succesfully exported 10 weight files to 'weights")

if __name__ == '__main__':
    main()