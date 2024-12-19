import numpy as np
import matplotlib.pyplot as plt

x_val = [i for i in range(1, 21)]

i1 = [102.0, 94.8, 98.3, 98.4, 102.0, 98.5, 99.0, 97.7, 100.0, 98.1]
i2 = [101.3, 98.7, 101.1, 98.4, 97.0, 96.7, 100.3, 101.4, 97.2, 101.0]
con_i = np.concatenate([i1, i2])

K = 1
H = 10
mu0 = np.mean(con_i)
cl = mu0
ucl = H
lcl = -(H)

y = con_i - mu0 - K
w = mu0 - con_i - K
#w = mu0 - con_i + K

c_plus = np.zeros(20)
c_plus[0] = max(y[0], 0)
for i in range(1, 20):
    if y[i] < 0:
        if (c_plus[i-1]+y[i])<=0:
            c_plus[i]=0
        elif (c_plus[i-1]+y[i])>0:
            c_plus[i] = y[i] + c_plus[i-1]
    elif y[i] >= 0:
        c_plus[i] =  y[i] + c_plus[i-1]
print(c_plus)

c_minus = np.zeros(20)
'''
c_minus[0] = min(w[0], 0)
for i in range(1, 20):
    if w[i] > 0:
        if (c_minus[i-1]+w[i])>=0:
            c_minus[i]=0
        elif (c_minus[i-1]+w[i])<0:
            c_minus[i] = w[i] + c_minus[i-1]
    elif w[i] <= 0:
        c_minus[i] =  w[i] + c_minus[i-1]
print(c_minus)
'''

c_minus[0] =  max(w[0], 0)
for i in range(1, 20):
    if w[i] < 0:
        if (c_minus[i-1]+w[i])<=0:
            c_minus[i]=0
        
        elif (c_minus[i-1]+w[i])>0:
            c_minus[i] = (w[i] + c_minus[i-1])
            
    elif w[i] >= 0:
        c_minus[i] =  (w[i] + c_minus[i-1])
print(c_minus)

print(len(c_minus))


plt.plot(x_val, c_plus, 'b^-', label='c+' )
plt.plot(x_val, -c_minus, 'bo-', label='c-' )
plt.axhline(0, color='green', linestyle='--',label='cl')
plt.axhline(ucl, color='r', linestyle='--',label='ucl')
plt.axhline(lcl, color='r', linestyle='--',label='lcl')

ooc_u = np.where(c_plus > ucl)[0]
plt.scatter(np.array(x_val)[ooc_u], c_plus[ooc_u], color='red', zorder=5)
ooc_u_indices = np.array(x_val)[ooc_u]

ooc_l = np.where(c_minus < lcl)[0]
plt.scatter(np.array(x_val)[ooc_l], c_minus[ooc_l], color='red', zorder=5)
ooc_l_indices = np.array(x_val)[ooc_l]

plt.title('CUSUM Chart')
plt.xlabel('Observation')
plt.ylabel('Cumulative Sum')
plt.legend(loc='best')
plt.grid(True)
plt.show()

print("Upper Control Limit (UCL) 이상점의 관측의 index:", ooc_u_indices)
print("Lower Control Limit (LCL) 이상점의 관측의 index:", ooc_l_indices)