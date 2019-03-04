import pandas as pd

class Neuron:
    def __init__(self):
        self.weight = [0 for i in range(12)]
        self.bias = 0

    def activate(self, data):
        value = sum([w * d for w, d in zip(self.weight, data)]) + self.bias
        teta = 0
        print(value)
        if value > teta:
            return 1
        else:
            return 0

    def train(self, datasets, epoch):
        for i in range(epoch):
            print("epoch", i+1)
            wold = [val for val in self.weight]
            bold = self.bias
            for data in datasets:
                # out = self.activate(data)

                alpha = 0.4
                if data[-1] != self.activate(data):
                # wi = wi + LR * xi * Error
                    self.weight = [w + alpha * xi * data[-1] for w, xi in zip(self.weight, data)]

                    self.bias = self.bias + alpha * data[-1]
                
                print(self.weight, self.bias)
            
            if self.weight == wold and self.bias == bold:
                return


df = pd.read_csv('data 4.csv', ';')
norm_df = (df - df.min())/(df.max()-df.min())
norm_df['target'] = df['target']

norm_df['target'] = norm_df['target'].where(norm_df['target'] == 2, 0)
norm_df['target'] = norm_df['target'].where(norm_df['target'] == 0, 1)

perceptron = Neuron()

perceptron.train(norm_df.values, 10)
print(perceptron.activate(norm_df.values[0]))
print(perceptron.activate(norm_df.values[14]))
print(perceptron.activate(norm_df.values[57]), norm_df.values[57,-1])
