import numpy as np
from sympy import primerange
import math

class SBIBD():
    def __init__(self, GROUP):
        self.GROUP = np.array(GROUP, dtype=int)
        self.C_SBIBD = np.array([])
        self.C_rest = []
        self.v = 0
        self.k = 0

    def create_SBIBD(self):
        a = lambda k: k**2 + k + 1
        prime = [1] + list(primerange(0, 30))
        for i in range(len(prime)-1):
            if a(prime[i]) <= len(self.GROUP) and a(prime[i+1]) > len(self.GROUP):
                self.k = prime[i]
                self.v = a(self.k)

        #creat the C_rest and C_SBIBD
        if len(self.GROUP) - self.v == 0:
            par_C_rest = np.random.choice(self.GROUP, size = 1)
            self.C_SBIBD = self.GROUP
            self.C_rest.append(par_C_rest.tolist())

        else:
            par_C_rest = np.random.choice(self.GROUP, size = len(self.GROUP) - self.v,
                                                  replace = False)
            self.C_SBIBD = self.GROUP[~np.isin(self.GROUP, par_C_rest)]
            self.C_rest.append(par_C_rest.tolist())

        return (self.C_rest[0], self.C_SBIBD, self.v, self.k)

#create the blocks
def create_blocks(SBIBD, a, k):
    blocks = []
    C = np.zeros((k+1, k+1), dtype=np.int32)
    D = np.zeros((a-k-1, k+1), dtype=np.int32)
    assert len(SBIBD) >= 3
    if len(SBIBD) == 3:
        blocks.append([SBIBD[0], SBIBD[1]])
        blocks.append([SBIBD[1], SBIBD[2]])
        blocks.append([SBIBD[2], SBIBD[0]])
    else:
        for i in range(k+1):
            for j in range(k+1):
                if j == 0:
                    C[i, j] = SBIBD[0]
                else:
                    C[i,j] = SBIBD[i*k+j]
            blocks.append(C[i,:].tolist())

        for i in range(k**2):
            for j in range(k+1):
                if j == 0:
                    t = int(math.floor(i/k) + 1)
                    D[i, j] = C[0, t].astype(int)
                else:
                    t = int((i+j*math.floor(i/k))%k +1)
                    D[i, j] = C[j, t].astype(int)
            blocks.append(D[i,:].tolist())

    return blocks





