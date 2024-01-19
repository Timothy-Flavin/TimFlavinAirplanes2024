import numpy as np
import matplotlib.pyplot as plt


r1 = 10
r2 = 12
x = np.arange(0,r1,0.2)
x2 = np.arange(0,r2,0.2)
inds = 4*np.exp(x)
inds2 = np.exp(x2)


plt.plot(inds, color="red",label=f"Inds 1 range: {r1}")
plt.plot(inds2, color="blue",label=f"Inds 2 range: {r2}")
plt.legend()

plt.show()

