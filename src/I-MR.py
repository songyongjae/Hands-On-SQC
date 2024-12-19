import numpy as np
import matplotlib.pyplot as plt

d2 = 1.128
D4 = 3.267
D3 = 0.000

x_val_r = [i for i in range(2, 21)]
x_val_i = [i for i in range(1, 21)]

i1 = [102.0, 94.8, 98.3, 98.4, 102.0, 98.5, 99.0, 97.7, 100.0, 98.1]
i2 = [101.3, 98.7, 101.1, 98.4, 97.0, 96.7, 100.3, 101.4, 97.2, 101.0]
con_i = np.concatenate([i1, i2])

R = np.abs(np.diff(con_i))

cl_r = np.mean(R)
ucl_r = D4 * cl_r
lcl_r = D3 * cl_r

plt.plot(x_val_r, R, 'bo-', label='MR' )
plt.axhline(y=cl_r, color='green', linestyle='--',label='cl')
plt.axhline(y=ucl_r, color='red', linestyle='--',label='ucl')
plt.axhline(y=lcl_r, color='red', linestyle='--',label='lcl')

plt.text(len(x_val_r) - 1, cl_r, f'{cl_r:.2f}', ha='right', va='bottom', color='blue')
plt.text(len(x_val_r) - 1, ucl_r, f'{ucl_r:.2f}', ha='right', va='bottom', color='red')
plt.text(len(x_val_r) - 1, lcl_r, f'{lcl_r:.2f}', ha='right', va='bottom', color='red')

ooc_u = np.where(R > ucl_r)[0]
plt.scatter(np.array(x_val_r)[ooc_u], R[ooc_u], color='red', zorder=5)

ooc_l = np.where(R < lcl_r)[0]
plt.scatter(np.array(x_val_r)[ooc_l], R[ooc_l], color='red', zorder=5)

plt.title('I-MR Chart')
plt.xlabel('Observation')
plt.ylabel('MR')
plt.legend(loc='upper center')
plt.grid(True)
plt.show()

cl_i = np.mean(con_i)
ucl_i = cl_i + 3 * cl_r / d2
lcl_i = cl_i - 3 * cl_r / d2

plt.plot(x_val_i, con_i, 'bo-', label='I' )
plt.axhline(y=cl_i, color='green', linestyle='--',label='cl')
plt.axhline(y=ucl_i, color='red', linestyle='--',label='ucl')
plt.axhline(y=lcl_i, color='red', linestyle='--',label='lcl')

plt.text(len(x_val_i) - 1, cl_i, f'{cl_i:.2f}', ha='right', va='bottom', color='blue')
plt.text(len(x_val_i) - 1, ucl_i, f'{ucl_i:.2f}', ha='right', va='bottom', color='red')
plt.text(len(x_val_i) - 1, lcl_i, f'{lcl_i:.2f}', ha='right', va='bottom', color='red')

ooc_u = np.where(con_i > ucl_i)[0]
plt.scatter(np.array(x_val_i)[ooc_u], con_i[ooc_u], color='red', zorder=5)

ooc_l = np.where(con_i < lcl_r)[0]
plt.scatter(np.array(x_val_i)[ooc_l], con_i[ooc_l], color='red', zorder=5)

plt.title('I-MR Chart')
plt.xlabel('Observation')
plt.ylabel('Individual')
plt.legend(loc='upper center')
plt.grid(True)
plt.show()