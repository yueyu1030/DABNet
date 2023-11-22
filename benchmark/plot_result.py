import json 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import numpy as np 

with open("results/acc.json", 'r') as f:
    acc = json.load(f)

with open("results/auc.json", 'r') as f:
    auc = json.load(f)

acc = np.array(acc)[1:]
auc = np.array(auc)[1:]

# acc_left = np.array([acc[0]] + list(acc)[:-1])
# acc_right = np.array(list(acc)[1:] + [acc[-1]])
plt.figure(figsize = [10, 6], dpi = 60)
plt.subplot(1, 2, 1)
plt.plot(acc - 0.01)
plt.axhline(0.808, 0, 200, linestyle = ':', c = 'gray')
plt.legend(["CausalGen", "FBNetGen"])
# plt.plot(smooth_acc-0.01)
plt.subplot(1, 2, 2)
plt.plot(auc - 0.01)
plt.axhline(75.0, 0, 200, linestyle = ':', c = 'gray')
plt.legend(["CausalGen", "FBNetGen"])
plt.savefig("results/acc.pdf")


