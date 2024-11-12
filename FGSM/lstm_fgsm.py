import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import json

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
    
def fgsm_perturbe(data, target, model, epsilon):
    loss_fn = torch.nn.NLLLoss()
    
    data.requires_grad = True
    outputs = model(data)
    loss = loss_fn(outputs, target)
    model.zero_grad()
    loss.backward()
    
    perturbed_data = data + epsilon * data.grad.data.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1) # 确保不会超出范围
    
    return perturbed_data

def plot_perturbed_images(model, test_loader, epsilons):
    examples = []
    
    for epsilon in epsilons:
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            perturbed_data = fgsm_perturbe(data, target, model, epsilon)
            examples.append((epsilon, perturbed_data[:10]))
            break
    
    plt.figure(figsize=(10, 10))
    for i, (epsilon, perturbed) in enumerate(examples):
        for j in range(10):
            plt.subplot(len(epsilons), 10, i * 10 + j + 1)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(f'eps: {epsilon}', fontsize=14)
            
            plt.imshow(perturbed[j].squeeze().detach().cpu().numpy(), cmap='gray')
    
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    test_data = MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    model = LSTMNet().to(device)
    model.load_state_dict(torch.load('model/mnist_lstm_model.pth', map_location=device, weights_only=True))
    
    model.eval()
    
    epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    accuracies = []
    
    plot_perturbed_images(model, test_loader, epsilons)
    
    for epsilon in epsilons:
        correct_cnt = 0
        total_cnt = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            perturbed_data = fgsm_perturbe(data, target, model, epsilon)
            outputs = model(perturbed_data)
            for i, output in enumerate(outputs):
                if torch.argmax(output) == target[i]:
                    correct_cnt += 1
                total_cnt += 1
                
        acc = correct_cnt / total_cnt
        accuracies.append(acc)
        print(f'epsilon: {epsilon:<5}, accuracy: {acc:.4f}')
        
    with open('FGSM/result.json', 'r') as file:
        data = json.load(file)
        
    data["lstm"] = {"epsilons": epsilons, "accuracies": accuracies}
    
    with open('FGSM/result.json', 'w') as f:
        json.dump(data, f, indent=4)
        
    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, accuracies, marker='o', linestyle='-')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    plt.title('')
    plt.grid(True)
    plt.show()