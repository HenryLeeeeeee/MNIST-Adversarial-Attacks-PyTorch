import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import json

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
    
    model = MLPNet().to(device)
    model.load_state_dict(torch.load('model/mnist_mlp_model.pth', map_location=device, weights_only=True))
    
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
        
    data["mlp"] = {"epsilons": epsilons, "accuracies": accuracies}
    
    with open('FGSM/result.json', 'w') as f:
        json.dump(data, f, indent=4)

    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, accuracies, marker='o', linestyle='-')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    plt.title('')
    plt.grid(True)
    plt.show()