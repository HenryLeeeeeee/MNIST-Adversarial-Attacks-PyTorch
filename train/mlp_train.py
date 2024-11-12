import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

class MLPNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
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
    loss_fn = torch.nn.NLLLoss()
    model.to(device)
    model.train()
    for epoch in range(epochs):
        # model.train()
        for data, target in train_dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()
        acc = test(model, test_dataloader, device)
        print(f'Epoch {epoch}, accuracy: {acc}')
        
    torch.save(model.state_dict(), 'model/mnist_model.pth')
    print('Model saved')

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f'Using device: {device}')

    train_loader, test_loader = load_data()
    model = MLPNet()
    train(model, train_loader, test_loader, device)

    # Visualize the model's prediction
    model.load_state_dict(torch.load('model/mnist_model.pth', weights_only=True))
    model.to(device)
    model.eval()
    with torch.no_grad():
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        output = model(data)
        for i in range(5):
            plt.imshow(data[i].cpu().squeeze().numpy(), cmap='gray')
            plt.title(f'Prediction: {torch.argmax(output[i]).item()}, True: {target[i].item()}')
            plt.show()
