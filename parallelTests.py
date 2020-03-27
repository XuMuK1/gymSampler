import numpy as np
import multiprocessing as mp
import scipy as sp
import matplotlib.pyplot as plt


def sample(seed):
    np.random.seed(seed) #critical!!!!
    return np.random.uniform()


pool = mp.Pool(mp.cpu_count()-1)
seed0=100
seed1=1e6
seedN=100000
seedH= (seed1-seed0)/(seedN-1)
result=pool.map(sample, 1035*np.arange(seed0,seed1+seedH/2,seedH).astype("int64"))
#print("result",result)
pool.close()


plt.hist(result)
plt.show()
