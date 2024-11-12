import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

class LSTMNet(torch.nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), x.size(2), x.size(3))
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        return out
    
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
    loss_fn = torch.nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    for epoch in range(epochs):
        for data, target in train_dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, target)
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
    
    model = LSTMNet().to(device)
    
    train(model, train_loader, test_loader, device)
    
    torch.save(model.state_dict(), 'model/mnist_lstm_model.pth')
    print('Model saved')