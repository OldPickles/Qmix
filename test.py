import matplotlib.pyplot as plt
import os
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.plot(x, y)

plt.show()

save_path = os.path.join(".", "test")
if not os.path.exists(save_path):
    os.makedirs(save_path)

plt.savefig(os.path.join(save_path, "test.png"))
plt.close()
