
import pywt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# dataset = "http://paos.colorado.edu/research/wavelets/wave_idl/sst_nino3.dat"
# df_nino = pd.read_table(dataset)
df_nino = pd.read_csv('sst_nino3.dat.txt', header=None)
N = df_nino.shape[0]
t0 = 1871
dt = 0.25
time = np.arange(0, N) * dt + t0
signal = df_nino.values.squeeze()

# PyWavelets offers two different ways to deconstruct a signal.

# (1) We can either apply pywt.dwt() on a signal to retrieve the approximation coefficients. Then apply the DWT on the
# retrieved coefficients to get the second level coefficients and continue this process until you have reached the
# desired decomposition level.

(cA1, cD1) = pywt.dwt(signal, 'db2', 'smooth')
reconstructed_signal = pywt.idwt(cA1, cD1, 'db2', 'smooth')

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(signal, label='signal')
ax.plot(reconstructed_signal, label='reconstructed signal', linestyle='--')
ax.legend(loc='upper left')
plt.show()

# (2) Or we can apply pywt.wavedec() directly and retrieve all of the the detail coefficients up to some level n. This
# functions takes as input the original signal and the level n and returns the one set of approximation coefficients
# (of the n-th level) and n sets of detail coefficients (1 to n-th level).

coeffs = pywt.wavedec(signal, 'db2', level=8)
reconstructed_signal = pywt.waverec(coeffs, 'db2')

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(signal[:1000], label='signal')
ax.plot(reconstructed_signal[:1000], label='reconstructed signal', linestyle='--')
ax.legend(loc='upper left')
ax.set_title('de- and reconstruction using wavedec()')
plt.show()