import scipy
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


def butter_bandpass(lowcut, highcut, fs, order=5, label=None):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    return sos


center_freqs = np.array([0.5 * 2 ** (n * 1 / 3.) for n in range(0, 29)])
center_freqs.sort()
lower_freqs = 2. ** (-1 / 6.) * center_freqs

t = np.linspace(0.0, 64, 1024)
s = np.sin(10 * np.pi * t)

fig, ax = plt.subplots()
plt.plot(t, s)

for lf in lower_freqs:
    sos = butter_bandpass(lf, lf * 2 ** (1/3.), fs=1000, order=2)
    output = signal.sosfilt(sos, s)
    plt.plot(t, output)

plt.show()




