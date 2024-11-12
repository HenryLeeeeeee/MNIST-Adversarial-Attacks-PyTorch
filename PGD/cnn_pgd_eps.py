import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import json

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
    
class PGDAttack(torch.nn.Module):
    def __init__(self, model, epsilon, alpha, iters, random_start=True):
        super().__init__()
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters
        self.random_start = random_start
        
    def forward(self, data, label):            
        loss_fn = torch.nn.CrossEntropyLoss()
        
        adv_data = data.clone().detach()
        
        if self.random_start:
            adv_data = adv_data + torch.empty_like(adv_data).uniform_(-self.epsilon, self.epsilon)
            adv_data = torch.clamp(adv_data, 0, 1).detach()
            
        for _ in range(self.iters):
            adv_data.requires_grad = True
            outputs = self.model(adv_data)
            
            cost = loss_fn(outputs, label)
            
            grad = torch.autograd.grad(cost, adv_data, retain_graph=False, create_graph=False)[0]

            adv_data = adv_data.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_data - data, min=-self.epsilon, max=self.epsilon)
            adv_data = torch.clamp(data + delta, 0, 1).detach()
            
        return adv_data
    
def plot_perturbed_images(model, test_loader, epsilons):
    examples = []
    
    for epsilon in epsilons:
        pgd_attack = PGDAttack(model, epsilon, 0.008, 40)
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            adv_data = pgd_attack(data, target)
            examples.append((epsilon, adv_data[:10]))
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
    
    model = CNNNet().to(device)
    model.load_state_dict(torch.load('model/mnist_cnn_model.pth', map_location=device, weights_only=True))
    
    # pgd_attack = PGDAttack(model, 8/255, 2/255, 40)
    
    model.eval()
    
    epsilons = [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    accuracies = []
    
    plot_perturbed_images(model, test_loader, epsilons=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
    
    for epsilon in epsilons:
        pgd_attack = PGDAttack(model, epsilon, 0.008, 40)
        correct_cnt = 0
        total_cnt = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            perturbed_data = pgd_attack(data, target)
            outputs = model(perturbed_data)
            for i, output in enumerate(outputs):
                if torch.argmax(output) == target[i]:
                    correct_cnt += 1
                total_cnt += 1
        acc = correct_cnt / total_cnt
        accuracies.append(acc)
        print(f'epsilon: {epsilon:<5}, accuracy: {acc:.4f}')
        
    with open('PGD/result.json', 'r') as file:
        data = json.load(file)
        
    data["cnn"] = {"epsilons": epsilons, "accuracies": accuracies}
    
    with open('PGD/result.json', 'w') as f:
        json.dump(data, f, indent=4)
        
    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, accuracies, marker='o', linestyle='-')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    plt.title('')
    plt.grid(True)
    plt.show()