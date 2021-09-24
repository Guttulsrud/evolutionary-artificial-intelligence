import numpy as np

ca = np.zeros((10, 10))
for x in ca:
    for y in x:
        ca[int(x)][int(y)] = 1

print(ca)
