
import pywt
import matplotlib.pyplot as plt
import numpy as np

# Analyzing the signal on different scales is also known as multiresolution / multiscale analysis, and decomposing your
# signal in such a way is also known as multiresolution decomposition, or sub-band coding.

x = np.linspace(0, 1, num=2048)
chirp_signal = np.sin(250 * np.pi * x ** 2)

fig, ax = plt.subplots(figsize=(6, 1))
ax.set_title("Original Chirp Signal: ")
ax.plot(chirp_signal)
plt.show()

data = chirp_signal
waveletname = 'sym5'

fig, axarr = plt.subplots(nrows=5, ncols=2, figsize=(6, 6))
for ii in range(5):
    (data, coeff_d) = pywt.dwt(data, waveletname)
    axarr[ii, 0].plot(data, 'r')
    axarr[ii, 1].plot(coeff_d, 'g')
    axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14, rotation=90)
    axarr[ii, 0].set_yticklabels([])
    if ii == 0:
        axarr[ii, 0].set_title("Approximation coefficients", fontsize=14)
        axarr[ii, 1].set_title("Detail coefficients", fontsize=14)
    axarr[ii, 1].set_yticklabels([])
plt.tight_layout()
plt.show()

# We can also use 'pywt.wavedec()' to immediately calculate the coefficients of a higher level. This functions takes as
# input the original signal and the level n and returns the one set of approximation coefficients (of the n-th level)
# nd n sets of detail coefficients (1 to n-th level).
