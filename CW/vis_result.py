import matplotlib.pyplot as plt
import json

with open('CW/result.json', 'r') as file:
    data = json.load(file)

fig, ax = plt.subplots()

colors = ['r', 'g', 'b', 'c']
labels = ['CNN', 'MLP', 'LSTM', 'VGG']

for i, (model, color) in enumerate(zip(data, colors)):
    epsilons = data[model]["cs"]
    accuracies = data[model]["accuracies"]
    ax.plot(epsilons, accuracies, color=color, label=model, marker='.')

ax.legend()

ax.set_title('')
ax.set_xlabel('c')
ax.set_ylabel('Accuracy')

plt.show()