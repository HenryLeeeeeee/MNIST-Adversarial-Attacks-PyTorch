import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*7*7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10),
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # print(x)
        return x
    
class CWAttack(torch.nn.Module):
    def __init__(self, model, device, c=1, kappa=0, steps=200, lr=0.01):
        super().__init__()
        self.model = model
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.device = device

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = torch.nn.MSELoss(reduction="none")
        Flatten = torch.nn.Flatten()

        optimizer = torch.optim.Adam([w], lr=self.lr)

        for step in range(self.steps):
            adv_images = self.tanh_space(w)

            current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = self.model(adv_images)
            f_loss = self.f(outputs, labels).sum()

            cost = L2_loss + self.c * f_loss
            
            # print(f'cost: {cost}, step: {step}, L2_loss: {L2_loss}, f_loss: {f_loss}')

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            pre = torch.argmax(outputs.detach(), 1)
            condition = (pre != labels).float()
            # print(condition)

            mask = condition * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            if step % max(self.steps // 10, 1) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()
        diff = (best_adv_images - images).abs().max().item()
        # print(f'diff: {diff:.4f}')
        return best_adv_images

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        return self.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def f(self, outputs, labels):
        one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]

        other = torch.max((1 - one_hot_labels) * outputs, dim=1)[0]
        real = torch.max(one_hot_labels * outputs, dim=1)[0]
        
        # print(f'real: {real}, other: {other}, outputs: {outputs}')

        return torch.clamp(real - other, min=-self.kappa)
    
def plot_perturbed_images(model, test_loader, cs):
    examples = []
    
    for data, target in test_loader:
        data = data.to(device)
        examples.append((0, data[:10]))
        break
    
    for c in cs:
        pgd_attack = CWAttack(model, device, c=c,lr=0.01)
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            adv_data = pgd_attack(data, target)
            examples.append((c, adv_data[:10]))
            break
    
    plt.figure(figsize=(10, 10))
    for i, (c, perturbed) in enumerate(examples):
        for j in range(10):
            plt.subplot(len(cs) + 1, 10, i * 10 + j + 1)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                if i == 0:
                    plt.ylabel('Original', fontsize=14)
                else:
                    plt.ylabel(f'c: {c}', fontsize=14)
            
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
    
    model = VGGNet().to(device)
    model.load_state_dict(torch.load('model/mnist_vgg_model.pth', map_location=device, weights_only=True))
    
    # pgd_attack = PGDAttack(model, 8/255, 2/255, 40)
    
    model.eval()
    
    cs = [0.1, 0.5, 1, 1.5, 2, 2.5, 3] 
    accuracies = []
    
    plot_perturbed_images(model, test_loader, cs=cs)
    
    for c in cs:
        cw_attack = CWAttack(model, device, c=c,lr=0.01)
        correct_cnt = 0
        total_cnt = 0
        for data, target in tqdm(test_loader, desc=f"Testing with c={c:<4}"):
            data, target = data.to(device), target.to(device)
            perturbed_data = cw_attack(data, target)
            outputs = model(perturbed_data)
            for i, output in enumerate(outputs):
                if torch.argmax(output) == target[i]:
                    correct_cnt += 1
                total_cnt += 1
        acc = correct_cnt / total_cnt
        accuracies.append(acc)
        print(f'c: {c:<5}, accuracy: {acc:.4f}')
        
    with open('CW/result.json', 'r') as file:
        data = json.load(file)
        
    data["vgg"] = {"cs": cs, "accuracies": accuracies}
    
    with open('CW/result.json', 'w') as f:
        json.dump(data, f, indent=4)
    
    plt.figure(figsize=(8, 6))
    plt.plot(cs, accuracies, marker='o', linestyle='-')
    plt.xlabel('c')
    plt.ylabel('Accuracy')
    plt.title('')
    plt.grid(True)
    plt.show()