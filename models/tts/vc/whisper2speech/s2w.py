import numpy as np
import soundfile as sf
from scipy.signal import lfilter, windows
from scipy.linalg import solve_toeplitz
import matplotlib.pyplot as plt
import math
import time

def autocorrelation(x, l):
    N = len(x)
    return sum(x[i] * x[i + l] for i in range(N - l))

    # full_corr = np.correlate(x, x, mode='full')
    # result_index = len(x) - 1 + l

    # return full_corr[result_index]

def levinson_durbin(x, order):
    r = np.array([autocorrelation(x, i) for i in range(order + 1)])
    a = np.zeros(order + 1)
    a[0] = 1.0
    if r[0] == 0:
        return a, 0
    a[1] = -r[1] / r[0]
    E = r[0] + r[1] * a[1]
    for k in range(1, order):
        lambda_ = -np.dot(a[:k + 1], r[1:k + 2][::-1]) / E
        a[:k + 2] += lambda_ * a[k+1::-1]
        E *= 1 - lambda_ ** 2
    return a, E

def s2w(input_file, frame_ms=20, lpc_order=None, rate=0.0, window_type='hamming', lpf=0.7, mode="read", samplerate=None):
    # t0 = time.time()
    # Load audio data
    if mode=='read':
        data, samplerate = sf.read(input_file)
        print("Speech duration:", len(data)/samplerate)
    elif mode=='wave':
        data = input_file
    else:
        raise NotImplementedError
    assert type(samplerate) is int
    
    # Ensure audio is mono
    if len(data.shape) > 1:
        data = data[:, 0]  # assuming the audio is stereo, take one channel

    # LPC order calculation
    if lpc_order is None:
        lpc_order = int(samplerate / 44100 * 40)

    # Frame length in samples
    frame_length = int(samplerate * frame_ms / 1000)
    if frame_length % 2 != 0:
        frame_length += 1

    # Window function selection
    if window_type == 'hamming':
        window = windows.hamming(frame_length)
    elif window_type == 'hanning':
        window = windows.hann(frame_length)
    elif window_type == 'blackman':
        window = windows.blackman(frame_length)
    else:
        raise ValueError("Unsupported window type")

    # Process each frame
    frames = len(data) // frame_length *2-1

    E = np.zeros(frame_length)
    v1 = np.zeros_like(data)
    v2 = np.zeros_like(data)

    for i in range(frames):
        start = i * frame_length // 2
        end = start + frame_length
        frame_data = data[start:end] * window
        a, e = levinson_durbin(frame_data, lpc_order)
        E = np.zeros(frame_length)
        for j in range(frame_length):
            E[j] = sum(a[n] * frame_data[j - n] for n in range(lpc_order + 1) if j >= n)

        for j in range(frame_length):
            v2[j + i * frame_length // 2] += E[j] * 10.0

        max_energy = np.sqrt(3 * np.sum(E ** 2) / frame_length)
        # uniformly random
        y = rate * E + (1.0 - rate) * max_energy * (np.random.rand(frame_length) * 2.0 - 1.0)

        # normal distribution random
        # y = rate * E + (1.0 - rate) * max_energy * np.clip(-0.5, 0.5, np.random.normal(0, 0.3, frame_length))

        # t01 = time.time()
        for j in range(frame_length):
            for n in range(1, lpc_order + 1):
                if j >= n:
                    y[j] -= a[n] * y[j - n]
        v1[start:end] += y
        # t02 = time.time()
        # print("conv time", t02-t01)

    for i in range(1, len(v1)):
        v1[i] = v1[i] - v1[i-1]*lpf

    # t1 = time.time()
    # print("Process time:", t1-t0)

    return v1, v2, samplerate
