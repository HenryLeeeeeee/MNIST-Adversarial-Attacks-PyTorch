import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import numpy as np
from tqdm import tqdm

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
        adv_samples = []
        
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
            
            adv_samples.append(adv_data.clone().detach())
        
        return adv_samples

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
    
    epsilon = 0.15
    alpha = 0.00375
    steps = 60
    
    pgd_attack = PGDAttack(model, epsilon, alpha, steps)
    accuracies_per_iter = []
    
    correct_cnt_per_iter = [0] * steps
    total_cnt_per_iter = [0] * steps

    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        
        perturbed_data_list = pgd_attack(data, target)
        
        for iter_idx, perturbed_data in enumerate(perturbed_data_list):
            outputs = model(perturbed_data)
            for i, output in enumerate(outputs):
                if torch.argmax(output) == target[i]:
                    correct_cnt_per_iter[iter_idx] += 1
                total_cnt_per_iter[iter_idx] += 1

    accuracies_per_iter = [correct_cnt / total_cnt for correct_cnt, total_cnt in zip(correct_cnt_per_iter, total_cnt_per_iter)]
    print(accuracies_per_iter)
    
    with open('PGD/result_steps.json', 'r') as file:
        data = json.load(file)    

    data['lstm'] = accuracies_per_iter
    
    with open('PGD/result_steps.json', 'w') as f:
        json.dump(data, f, indent=4)


    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracies_per_iter) + 1), accuracies_per_iter, marker='.')
    plt.title('')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()