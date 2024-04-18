import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt

A = np.array([[0,1,0,0],
              [-2,-1,0,-1.5],
              [0,0,0,1],
              [.5,.5,0,-.1]])
B = np.array([[0],[-0.35],[0],[0]])
x0 = np.array([[5],[0],[0],[0]])

eigs = np.linalg.eig(A)
print(eigs) #imaginary > real part then it will oscilate make then = to get 4% overshoot damped fast
t=np.linspace(0,20,200)

vec = np.zeros((t.shape[0],4))
vec[0] = x0[:,0]
for i in range(t.shape[0]):
  vec[i] = (scipy.linalg.expm(A*t[i])@x0)[:,0]

plt.plot(vec)
plt.show()