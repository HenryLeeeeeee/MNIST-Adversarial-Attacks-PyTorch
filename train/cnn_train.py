import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

class CNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.fc1 = nn.Linear(20*10*10, 500)
        self.fc2 = nn.Linear(500, 10)
        
    def forward(self, x):
        in_size = x.size(0)
        
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        
        x = self.conv2(x)
        x = nn.functional.relu(x)
        
        x = x.view(in_size, -1)
        
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        
        return x
    
def load_data():
    train_data = MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
    test_data = MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    return train_loader, test_loader

def test(model, test_dataloader, device):
    model.eval()
    correct_cnt = 0
    total_cnt = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            for i, output in enumerate(outputs):
                if torch.argmax(output) == target[i]:
                    correct_cnt += 1
                total_cnt += 1
    return correct_cnt / total_cnt

def train(model, train_dataloader, test_dataloader, device, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        for data, target in train_dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = nn.functional.cross_entropy(outputs, target)
            loss.backward()
            optimizer.step()
        acc = test(model, test_dataloader, device)
        print(f'Epoch {epoch}, accuracy: {acc}')
        
if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    train_loader, test_loader = load_data()
    
    model = CNNNet().to(device)
    
    train(model, train_loader, test_loader, device)
    
    torch.save(model.state_dict(), 'model/mnist_cnn_model.pth')
    print('Model saved')