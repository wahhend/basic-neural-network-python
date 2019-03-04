class Neuron:
    def __init__(self):
        self.weight = [0, 0]
        self.bias = 0

    def activate(self, data):
        value = sum([w * d for w, d in zip(self.weight, data)]) + self.bias
        teta = 0.2
        # print(value)
        if value > teta:
            return 1
        elif value <= teta and value >= -teta:
            return 0
        else:
            return -1

    def train(self, datasets, epoch):
        for i in range(epoch):
            print("epoch", i+1)
            wold = [val for val in self.weight]
            bold = self.bias
            for data in datasets:
                out = self.activate(data)

                
                alpha = 1
                if data[-1] != self.activate(data):
                # wi = wi + LR * xi * Error
                    self.weight = [w + alpha * xi * data[-1] for w, xi in zip(self.weight, data)]

                    self.bias = self.bias + alpha * data[-1]
                print(self.weight, self.bias)
            if self.weight == wold and self.bias == bold:
                
                return

datasets = [
    [1, 1, 1],
    [1, 0, -1],
    [0, 1, -1],
    [0, 0, -1]
]

simple_slp = Neuron()

simple_slp.train(datasets, 11)
