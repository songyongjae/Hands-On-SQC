import numpy as np
import matplotlib.pyplot as plt

x_val = [i for i in range(1, 16)]

a1 = [1.0, 1.0, 1.0, 1.0]
a2 = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
a3 = [1.2, 1.2, 1.2, 1.2]
con_a = np.concatenate([a1, a2, a3])

x1 = [2, 5, 3, 2]
x2 = [1, 5, 2, 4, 2, 6, 4]
x3 = [1, 11, 3, 8]
con_x = np.concatenate([x1, x2, x3])

u = con_x / con_a
cl = (np.sum(con_x) - 11) / (np.sum(con_a) - 1.2)

plt.plot(x_val, u, 'bo-', label='u' )
plt.axhline(y=cl, color='green', linestyle='--',label='cl')
ucl = cl + 3 * np.sqrt(cl/con_a)
plt.step(x_val, ucl, color='r', linestyle='--',label='ucl')
lcl = np.maximum(cl - 3 * np.sqrt(cl / con_a), np.zeros_like(cl))
plt.step(x_val, lcl, color='r', linestyle='--',label='lcl')

plt.text(len(x_val), cl, f'{cl:.2f}', ha='left', va='center', color='blue')
plt.text(len(x_val), ucl[-1], f'{ucl[-1]:.2f}', ha='left', va='center', color='red')
plt.text(len(x_val), lcl[-1], f'{lcl[-1]:.2f}', ha='left', va='center', color='red')

out_of_control = np.where(u > ucl)[0]
plt.scatter(np.array(x_val)[out_of_control], u[out_of_control], color='red', zorder=5)

plt.title('U-Chart')
plt.xlabel('#sample')
plt.ylabel('u')
plt.legend(loc='best')
plt.grid(True)
plt.show()