import matplotlib.pyplot as plt
import numpy as np


def render(vector):
    vector_as_integers = np.array(vector, dtype=int)
    plt.imshow(vector_as_integers, vmin=0, vmax=1, cmap='BuGn')
    plt.axis('off')
    plt.show()
