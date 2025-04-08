import sys, os
# 親ディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simplenet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

net = simplenet()
print(net.W)

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simplenet()

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)