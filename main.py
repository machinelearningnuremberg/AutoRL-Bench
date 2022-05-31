import json
import matplotlib.pyplot as plt

file = "data/SpaceInvaders-v0_DQN_random_lr_-3-3-3_gamma_0.990.980.99_seed0_eval.json"
with open(file) as f:
    data = json.load(f)

print(data.keys())

fig, axis = plt.figure()
plt.plot(data["timestamps_train"], data["returns_train"])