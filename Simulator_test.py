import numpy as np
import matplotlib.pyplot as plt
from airplane_sim.plane1 import airplane_lrh

y=1

def blah(x1):
  x1 = x1+1
  x2 = 3
  wada = 5
  if x1<5:
    blah(x1)
  print("yeet")

blah(1)

airplane_lrh()
r1 = 10
r2 = 10
x = np.arange(0,r1,0.1)
x2 = np.arange(0,r2,0.1)
inds = x*x*x*x
inds2 = np.exp(x2)

plt.plot(inds, color="red",label=f"Inds 1 range: {r1}")
plt.plot(inds2, color="blue",label=f"Inds 2 range: {r2}")
plt.legend()

plt.show()

