# convert from 1 binary string to n decimal
# convert back
import numpy as np

b = np.array([1, 0, 1, 0, 1, 1, 1, 0])

d = []
divisor = 4
d = np.split(b, divisor)
#print(d)


for i in range(0, divisor):
    num = d[i].dot(1 << np.arange(d[i].size)[::-1])
    print(num)


    b = np.fromstring(np.binary_repr(num), dtype='S1').astype(int)
    print(b)

