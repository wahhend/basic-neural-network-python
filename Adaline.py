import numpy as np
import pandas as pd

class Adaline(object):
    def __init__(self,size):
        self.weight = [0 for i in range(size)]
        self.bias = 0
        # self.target_is_bipolar = False


    def train(self, datasets, threshold, alpha):
        max_error = threshold
        # Cek apakah target berupa bipolar atau bukan
        # if np.array_equiv(np.sort(np.unique(train_target)),[-1,1]) is True:
        #     self.target_is_bipolar = True
        # else:
        #     self.target_is_bipolar = False
        i = 0
        while max_error >= threshold:
            i+=1
            print("epoch", i)
            max_error = 0
            for data in datasets:
                target = data[-1]
                data = np.array(data[:-1])
                
                y = np.dot(self.weight, data) + self.bias
                delta_w = alpha * (target-y) * data
                self.bias = self.bias + alpha * (target-y)
                self.weight = self.weight + delta_w
                
                max_error = np.max(np.append(delta_w, max_error))

            print(self.weight, self.bias)

    def test(self, test_data):
        v = np.dot(self.weight, test_data) + self.bias
        print(v)
        return self.aktivasi_biner(v)

    def aktivasi_bipolar(self,x):
        if x < 0:
            return -1
        else:
            return 1

    def aktivasi_biner(self,x):
        if x < 0:
            return 0
        else:
            return 1

    def get_weight(self):
        return (self.weight,self.bias)


df = pd.read_csv('data 4.csv', ';')
norm_df = (df - df.min())/(df.max()-df.min())
norm_df['target'] = df['target']
norm_df['target'] = norm_df['target'].where(norm_df['target'] == 2, 0)
norm_df['target'] = norm_df['target'].where(norm_df['target'] == 0, 1)

adaline = Adaline(12)
adaline.train(norm_df.values[:19], 0.05, 0.1)

adaline.test(norm_df.values[1,:-1])
adaline.test(norm_df.values[10,:-1])
# adaline.test(norm_df.values[48,:-1])

# datasets = [
#     [1, 1, 1],
#     [1, -1, -1],
#     [-1, 1, -1],
#     [-1, -1, -1]
#     ]

# adaline = Adaline(2)
# adaline.train(datasets, 0.07)

# print(adaline.test([1, 1]))
# print(adaline.test([1, -1]))
# print(adaline.test([-1, 1]))
# print(adaline.test([-1, -1]))


# iris = pd.read_csv('iris.csv', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
# norm_iris = (iris - iris.min())/(iris.max()-iris.min()) 

# adaline = Adaline(4)

# adaline.train(norm_iris.values, 0.03, 0.1)

# adaline.test(norm_iris.values[28,:-1])
# adaline.test(norm_iris.values[65,:-1])