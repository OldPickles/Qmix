import matplotlib.pyplot as plt
import os
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.plot(x, y)

plt.show()

save_path = os.path.join("images", "test.png")

plt.savefig(save_path)
plt.close()
