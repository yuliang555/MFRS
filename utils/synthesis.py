import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from math import sin, pi
import random
from sklearn.preprocessing import StandardScaler



datapath = "./datasets"

L = 10000
C = 7

T1 = 720       # primary base pattern
T2 = T1 // 2   # harmonic base pattern
T3 = T1 // 3   # harmonic base pattern
T4 = T1 // 4   # harmonic base pattern

random_type = "gaus" # "gaus", "pois"
gaus_mean = 0
gaus_std = 1
pois_lam = 1

filename = f"gaus_mean={gaus_mean}_std={gaus_std}.csv" if random_type == "gaus" else f"pois_lam={pois_lam}.csv"



##################################### deterministic signal #######################################

amp1 = np.random.uniform(1.5, 2, C)
amp2 = np.random.uniform(1, 1.5, C)
amp3 = np.random.uniform(0.5, 1, C)
amp4 = np.random.uniform(0, 0.5, C)

deter_signal_1 = np.array([[a1 * sin(t * 2 * pi / T1) for t in range(L)] for a1 in amp1]).T
deter_signal_2 = np.array([[a2 * sin(t * 2 * pi / T2) for t in range(L)] for a2 in amp2]).T
deter_signal_3 = np.array([[a3 * sin(t * 2 * pi / T3) for t in range(L)] for a3 in amp3]).T
deter_signal_4 = np.array([[a4 * sin(t * 2 * pi / T4) for t in range(L)] for a4 in amp4]).T

deter_signal = deter_signal_1 + deter_signal_2 + deter_signal_3 + deter_signal_4




##################################### random signal #######################################

if random_type == "gaus":
    random_signal = gaus_std * np.random.randn(L, C) + gaus_mean
else:
    random_signal = np.random.poisson(lam=pois_lam, size=(L, C))



##################################### synthetic signal #######################################

syn_signal = deter_signal + random_signal











df = pd.DataFrame(syn_signal)
df.to_csv(f"{datapath}/syn_pbp={T1}_{filename}", index=None)








