import numpy as np
from scipy.signal import gaussian, convolve, find_peaks, welch, spectrogram
from scipy.fft import fft, ifft

def Brainard_features(y, fs):
    # Check if the input signal has enough samples
    if len(y) < 240:
        print('Not enough samples. min. is 240.')
        # return -1
        features = {}
        features['FF'] =  np.nan
        features['amplitude'] =  np.nan
        features['time_to_half_peak'] =  np.nan
        features['FF_slope'] =  np.nan
        features['Amplitude_Slope'] =  np.nan
        features['Spectral_Entropy'] =  np.nan
        features['Temporal_Entropy'] =  np.nan
        return features
    dt = 1 / fs  # time bin
    # Create a gaussian window of size ~3ms and width 2ms
    L = int(np.floor(0.00313 * fs))  # so that it's 150 in fs=48k
    sigma = 0.002 * fs
    alpha = (L - 1) / (2 * sigma)
    g = gaussian(150, alpha)
    g = g / np.sum(g)
    rect_y = convolve(y ** 2, g, mode='same')  # smoothed rectified signal
    win_size = int(np.floor(0.008 * fs))
    step_size = int(np.floor(0.002 * fs))
    # Calculate FF (Fundamental frequency) trace
    FF = []
    curr = 1
    while True:
        if curr + win_size + step_size - 1 > len(y):
            currwin = np.arange(curr, len(y))
        else:
            currwin = np.arange(curr, curr + win_size)
        a = np.correlate(y[currwin], y[currwin], mode='full')
        a = a[len(currwin) - 1:]
        a = np.concatenate(([a[0] + 1], a))
        a_peaks, _ = find_peaks(a)
        try:
            FF.append(1 / (a_peaks[0] * dt))
        except IndexError:
            pass
        curr += step_size
        if curr + win_size >= len(y):
            break
    features = {}
    if FF:
        if len(FF) > 1:
            features['FF'] = np.mean(FF[int(len(FF) * 0.1):int(len(FF) * 0.9)])
        else:
            features['FF'] = FF[0]
    else:
        features['FF'] = np.nan
        print('Could not calculate FF')
    features['amplitude'] = np.max(rect_y)
    # Time to half peak amplitude
    loc_half_peak = np.min(np.where(rect_y >= np.max(rect_y) / 2))
    features['time_to_half_peak'] = loc_half_peak * dt
    # Frequency slope
    if len(FF) > 1:
        if len(FF) > 2:
            features['FF_slope'] = np.mean(np.diff(FF[int(len(FF) * 0.1):int(len(FF) * 0.9)]))
        else:
            features['FF_slope'] = FF[1] - FF[0]
    else:
        features['FF_slope'] = np.nan
    # Amplitude slope
    P1 = np.mean(rect_y[int(len(y) * 0.1):int(len(y) * 0.5)])
    P2 = np.mean(rect_y[int(len(y) * 0.5):int(len(y) * 0.9)])
    features['Amplitude_Slope'] = (P1 - P2) / (P1 + P2)
    # Spectral entropy
    try:
        F,Pxx = welch(y[int(len(y) * 0.1):int(len(y) * 0.9)], fs=fs, nperseg=int(0.005 * fs), noverlap=int(0.0025 * fs))
    except ValueError:
        F,Pxx = welch(y[int(len(y) * 0.1):int(len(y) * 0.9)], fs=fs, nperseg=int(0.0035 * fs), noverlap=int(0.002 * fs))
    # Pxx = Pxx / np.sum(Pxx)
    # Pxx += np.finfo(float).eps
    features['Spectral_Entropy'] = -np.sum(np.log(Pxx+np.finfo(float).eps) * Pxx) / np.log(2)
    # Temporal entropy (use 30 bins)
    Pa, _ = np.histogram(rect_y[int(len(y) * 0.1):int(len(y) * 0.9)], bins=30)
    Pa = Pa / np.sum(Pa)
    features['Temporal_Entropy'] = -np.sum(np.log(Pa+np.finfo(float).eps) * Pa) / np.log(2)
    return features