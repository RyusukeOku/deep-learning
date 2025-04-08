import numpy as np
import matplotlib.pyplot as plt #pylab -> pyplot

def step_function(x):
    return np.array(x > 0, dtype = int) #dtype = int に変更

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()

print(step_function(np.array([-1.0, 1.0, 2.0])))