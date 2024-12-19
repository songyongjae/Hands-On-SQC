import numpy as np
import matplotlib.pyplot as plt

lambd = 0.2
n = 1
x_val = [i for i in range(1, 21)]

i1 = [102.0, 94.8, 98.3, 98.4, 102.0, 98.5, 99.0, 97.7, 100.0, 98.1]
i2 = [101.3, 98.7, 101.1, 98.4, 97.0, 96.7, 100.3, 101.4, 97.2, 101.0]
con_i = np.concatenate([i1, i2])

d2=1.128
R = np.abs(np.diff(con_i))
sigma_hat = np.mean(R) / d2

z = np.zeros(20)
z[0] = con_i[0]

for index, xi in enumerate(con_i[1:], start=1):
    z[index] = lambd * xi + (1- lambd) * z[index-1]

cl = np.full(20, z[0])

var_z = np.zeros(20)
for i in range (1, 21):
    var_z[i-1] = ((sigma_hat ** 2 ) * lambd * (1 - (1-lambd)**(2*i))) / (n * (2-lambd))

ucl = cl + 3 * np.sqrt(var_z)
lcl = cl - 3 * np.sqrt(var_z)

plt.plot(x_val, z, 'bo-', label='z' )
plt.step(x_val, cl, color='green', linestyle='--',label='cl')
plt.step(x_val, ucl, color='r', linestyle='--',label='ucl')
plt.step(x_val, lcl, color='r', linestyle='--',label='lcl')

plt.text(len(x_val) - 1, cl[-1], f'{cl[-1]:.2f}', ha='right', va='bottom', color='blue')
plt.text(len(x_val) - 1, ucl[-1], f'{ucl[-1]:.2f}', ha='right', va='bottom', color='red')
plt.text(len(x_val) - 1, lcl[-1], f'{lcl[-1]:.2f}', ha='right', va='bottom', color='red')

ooc_u = np.where(z > ucl)[0]
plt.scatter(np.array(x_val)[ooc_u], z[ooc_u], color='red', zorder=5)
ooc_u_indices = np.array(x_val)[ooc_u]

ooc_l = np.where(z < lcl)[0]
plt.scatter(np.array(x_val)[ooc_l], z[ooc_l], color='red', zorder=5)
ooc_l_indices = np.array(x_val)[ooc_l]


plt.title('EWMA Chart')
plt.xlabel('Observation')
plt.ylabel('z')
plt.legend(loc='best')
plt.grid(True)
plt.show()

print("Upper Control Limit (UCL) 이상점의 관측의 index:", ooc_u_indices)
print("Lower Control Limit (LCL) 이상점의 관측의 index:", ooc_l_indices)