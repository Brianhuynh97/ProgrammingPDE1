import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, csc_matrix
from scipy.sparse.linalg import spsolve

x1 = np.linspace(0, 1, 1000)
x2 = np.linspace(1, 2, 1000)
fig = plt.figure()
plt.plot(x1, 1-x1, color='blue', label='$\phi_0$')
plt.plot(x1, x1,color='green', label='$\phi_1$')
plt.plot(x2, 2-x2, color='green', label='$\phi_1$')
plt.plot(x2, x2-1, color='orange', label='$\phi_2$')
plt.axvline(x=1, color='blue', linestyle=':')
plt.axvline(x=2, color='blue', linestyle=':')
plt.margins(x=0)
plt.margins(y=0)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()