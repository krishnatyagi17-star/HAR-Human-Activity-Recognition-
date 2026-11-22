import numpy as np
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

def get_features_v2(window):
    feats = []

    mags = []
    for start in [0, 3, 6]:
        mags.append(np.sqrt((window[:, start:start + 3] ** 2).sum(axis=1)))
    mags = np.array(mags).T
    full = np.hstack([window, mags])

    for c in range(full.shape[1]):
        col = full[:, c]
        feats.append(col.mean())
        feats.append(col.std())
        feats.append(col.min())
        feats.append(col.max())
        feats.append(np.median(col))
        feats.append(np.sum(col ** 2) / len(col))
        feats.append(np.percentile(col, 75) - np.percentile(col, 25))
        feats.append(skew(col))
        feats.append(kurtosis(col))
        spectrum = np.abs(np.fft.rfft(col - col.mean()))
        feats.append(np.argmax(spectrum))
        feats.append(np.sum(spectrum ** 2) / len(spectrum))

    for start in [0, 3, 6]:
        feats.append(np.corrcoef(window[:, start], window[:, start + 1])[0, 1])
        feats.append(np.corrcoef(window[:, start], window[:, start + 2])[0, 1])
        feats.append(np.corrcoef(window[:, start + 1], window[:, start + 2])[0, 1])

    return np.nan_to_num(feats)
