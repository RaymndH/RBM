import numpy as np
import matplotlib.pyplot as plt
xax=[.185, .5, .75, .88, 1.12]
y1=[2.3e8, 2.7e5, 2.5e5, 1.5e4, 5.9e3]
y2=[np.nan, np.nan, 3.3e5, 9.1e6, 6.7e3]
plt.scatter(xax,y1, label='CAC')
plt.scatter(xax,y2, label='RBM')
plt.xlabel("alpha")
plt.ylabel("TTS")
plt.yscale("log")
plt.legend()
plt.show()
