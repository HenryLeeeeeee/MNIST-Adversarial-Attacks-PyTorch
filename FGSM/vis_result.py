import matplotlib.pyplot as plt
import json

with open('FGSM/result.json', 'r') as file:
    data = json.load(file)

fig, ax = plt.subplots()

colors = ['r', 'g', 'b', 'c']
labels = ['CNN', 'MLP', 'LSTM', 'VGG']

for i, (model, color) in enumerate(zip(data, colors)):
    epsilons = data[model]["epsilons"]
    accuracies = data[model]["accuracies"]
    ax.plot(epsilons, accuracies, color=color, label=labels[i], marker='.')

ax.legend()

ax.set_title('Accuracy vs Epsilon for Different Networks')
ax.set_xlabel('Epsilon')
ax.set_ylabel('Accuracy')

plt.show()