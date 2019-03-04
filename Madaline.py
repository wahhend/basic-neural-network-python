import numpy as np
import pandas as pd


class Adaline:
    def __init__(self, size):
        self.weight = np.zeros(size)
        self.bias = 0

    def activate_bip(self, x):
        if x > 0:
            return 1
        else:
            return -1
    
    def output(self, data):
        # print(len(self.weight), len(data))
        return np.dot(self.weight, data) + self.bias


class Madaline:
    def __init__(self, size):
        self.hiddens = [Adaline(size), Adaline(size)]
        self.output = Adaline(2)
        self.output.weight = [0.5, 0.5]
        self.output.bias = 0


    def train(self, datasets, alpha):
        for data in datasets:
            target = data[-1]
            data = data[:-1]
            zin = [hidden.output(data) for hidden in self.hiddens]
            z = [hidden.activate_bip(zi) for zi, hidden in zip(zin, self.hiddens)]
            print("z", z)
            y = self.output.activate_bip(self.output.output(z))
            print("y", y)

            if target != y:
                if target == 1:
                    idx = list(np.abs(zin)).index(np.amin(np.abs(zin)))
                    neuron_to_update = self.hiddens[idx]
                    neuron_to_update.bias = neuron_to_update.bias + alpha * (1 - zin[idx])
                    delta_w = alpha * (1 - zin[idx]) * data
                    neuron_to_update.weight = neuron_to_update.weight + delta_w
                else:
                    for hidden, zi in zip(self.hiddens, zin):
                        if zi > 0:
                            hidden.bias = hidden.bias + alpha * (-1 - zi)
                            delta_w = alpha * (-1 - zi) * data
                            hidden.weight = hidden.weight + delta_w
    
    def test(self, data):
        z = [hidden.activate_bip(hidden.output(data)) for hidden in self.hiddens]
        print("z", z)
        y = self.output.activate_bip(self.output.output(z))
        print("y", y)            
        return y


iris = pd.read_csv('iris.csv', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
norm_iris = (iris - iris.min())/(iris.max()-iris.min()) 

madaline = Madaline(4)

for i in range(10):
    madaline.train(norm_iris.values, 0.1)

print(madaline.test(norm_iris.values[18,:-1]))
print(madaline.test(norm_iris.values[28,:-1]))
print(madaline.test(norm_iris.values[48,:-1]))
print(madaline.test(norm_iris.values[65,:-1]))
print(madaline.test(norm_iris.values[90,:-1]))
print(madaline.test(norm_iris.values[75,:-1]))