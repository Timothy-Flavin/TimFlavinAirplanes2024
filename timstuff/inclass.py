import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt
import control

A = np.array([[0,1,0,0],
              [-2,-1,0,-1.5],
              [0,0,0,1],
              [.5,.5,0,-.1]])
B = np.array([[0],[-0.35],[0],[0]])
C = np.array([[0,1,0,1],[1,0,1,0]]) #wrong numbers
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

observability = np.linalg.matrix_rank(control.obsv(A,C))
poles = [-1, -3, -2+2j,-2-2j]#-2+2j, -2-2j]
# Poles picked in class the first pole -1 is the dominant one because E^-1t is bigger than e^-3t
# Making the imaginary bigger causes a faster response but less damped
k = control.acker(A,B,poles) # acker is like place from matlab. it is one method
newA = A-B*k

vec = np.zeros((t.shape[0],4))
vec[0] = x0[:,0]
for i in range(t.shape[0]):
  vec[i] = (scipy.linalg.expm(newA*t[i])@x0)[:,0]

plt.plot(vec)
plt.legend(["B","B_dot","Chi","Chi_dot"])
plt.show()


# Controls stuff we talked about
# Chi_dot = AX+BU
# We need to translate A and B into the digital 
# domain because we are in continuous S domain
# X[n+1] = A_d[n] + B_dU[n]
# A_d is discrete version of A I think
# Here we are looking for the state X not X_dot
ts=0.1 #timestep
D = np.zeros((2,1))

continuous_system = control.ss(A,B,C,D)
discrete_system = control.c2d(continuous_system,ts)

Ad = discrete_system.A
Bd = discrete_system.B
Cd = discrete_system.C

print(discrete_system)
x = x0
xA = np.zeros((t.shape[0],4))
#X = Ad*X + Bd*U
U=-k@x
for i in range(t.shape[0]):
  xA[i] = x[:,0]
  x = Ad@x + Bd*U
  U=-k@x

plt.plot(t,xA)
plt.show()



# Notes from monday
#   U-->[Chi = Ax + By, y=Cx]-->y
Ts=0.02
dis_sys = control.c2d(control.ss(A,B,C,D),Ts,'zoh')
Ad = dis_sys.A
Bd = dis_sys.B

# Design the full state observer
# real part of poles needs to go to zero 5-10x quicker than the plan moves
obsv_poles = np.array([-10,-30,-20+20j, -20-20j])
obsv_poles = np.exp(obsv_poles*Ts) # z to s transform memorize it?
G = control.place(Ad.T,C.T,obsv_poles)
print(G)
# simulate
XA = []
YA = []
XA_hat = []
YA_hat = []
X = x0
Y = C@X
X_hat = x0
Y_hat = C@X
Ad_e = (np.random.rand(A.shape[0],A.shape[1])*2-1)*0.01 + Ad
t=np.linspace(0,20,int(20/Ts))
for ti in t:  
  XA.append(X)
  YA.append(Y)
  XA_hat.append(X_hat)
  YA_hat.append(Y_hat)
  u=0
  X = Ad@X + Bd*u
  X_hat = Ad_e@X_hat+Bd*u
  Y = C@X
  Y_hat = C@X_hat #need to add g somehow

plt.plot(np.array(XA)[:,:,0])
plt.plot(np.array(XA_hat)[:,:,0],linestyle='dashed')
plt.legend(['beta','betad','chi','chid'])
plt.show()