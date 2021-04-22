import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataFrameA = pd.read_excel('data/train/A.xlsx', header=None)
dataFrameG = pd.read_excel('data/train/G.xlsx', header=None) 

dataA = np.array(dataFrameA.drop([0,1,2,3,4,5], axis=1))
dataG = np.array(dataFrameG.drop([0,1,2,3,4,5], axis=1))

plt.plot(dataA[100])
plt.show()

plt.plot(dataG[100])
plt.show()