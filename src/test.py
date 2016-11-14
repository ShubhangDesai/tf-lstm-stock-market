from __future__ import print_function
import numpy as np
import cStringIO
import predict

stock = 'AAPL'
# stock = 'GOOGL'
# stock = 'MSFT'

def generate_data(csv_file):
    with open(csv_file+'.csv', "r") as myfile:
        data = myfile.read().replace('"', '')
    data = np.genfromtxt(cStringIO.StringIO(data), delimiter=',')[1:, 1:]
    data = data.astype(float)
    data = data[1:] / data[:-1] - 1
    data = np.fliplr(data.transpose())
    data = data[:, -30:]
    return data

data = generate_data('../res/'+stock)
data = data.astype(np.float32)

print(predict.predict(stock, data))