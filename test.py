import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.01, 10)
y = 1/(x * 2) + x

plt.plot(x, y)
plt.savefig('artifacts/remote_test5.png')
