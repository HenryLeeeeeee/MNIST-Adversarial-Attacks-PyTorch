import matplotlib.pyplot as plt
import json

with open('PGD/result_steps.json', 'r') as file:
    data = json.load(file)

fig, ax = plt.subplots()

colors = ['r', 'g', 'b', 'c']
labels = ['CNN', 'MLP', 'LSTM', 'VGG']

for i, (model, color) in enumerate(zip(data, colors)):
    accuracies = data[model]
    steps = range(1, len(accuracies) + 1)
    ax.plot(steps, accuracies, color=color, label=model, marker='')

ax.legend()

ax.set_title('')
ax.set_xlabel('Step')
ax.set_ylabel('Accuracy')

plt.show()