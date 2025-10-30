import numpy as np

def gcc_phat(sig, refsig, fs=32000, max_tau=None, interp=16):
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n*interp)
    REFSIG = np.fft.rfft(refsig, n*interp)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.irfft(R / (np.abs(R)+1e-15))
    max_shift = int(interp * fs * max_tau) if max_tau else int(len(cc)/2)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp*fs)
    return tau

# use gcc_phat with mic spacing
def doa_azimuth(mics, fs=32000, mic_distance=0.08, c=343.0):
    tau = gcc_phat(mics[:,0], mics[:,1], fs, mic_distance/c)
    angle = np.degrees(np.arcsin(np.clip(tau*c/mic_distance, -1, 1)))
    return angle
