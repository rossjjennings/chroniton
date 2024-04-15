import numpy as np

def fft_roll(arr, shift):
    """
    Roll array by a given (possibly fractional) amount, in bins.
    Works by multiplying the FFT of the input array by exp(-2j*pi*shift*f)
    and Fourier transforming back. The sign convention matches that of
    numpy.roll() -- positive shift is toward the end of the array.
    This is the reverse of the convention used by pypulse.utils.fftshift().
    If the array has more than one axis, the last axis is shifted.
    """
    n = arr.shape[-1]
    if not hasattr(shift, 'shape'):
        shift = np.array(shift)
    shift = shift[..., np.newaxis]
    phase = -2j*np.pi*shift*np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(arr)*np.exp(phase), n)

def fft_interp(arr, x):
    """
    Interpolate the values in `arr` at the locations `x`, in bins.
    As with `fft_roll()`, this works by using the amplitudes and frequencies
    associated with the DFT of `arr` to define a continuous function.
    """
    n = arr.shape[-1]
    if not hasattr(x, 'shape'):
        x = np.array(x)
    phase = 2j*np.pi*x[..., np.newaxis]*np.fft.fftfreq(n)
    return np.mean(np.fft.fft(arr)*np.exp(phase), axis=-1)[()]

def lerp(arr, x):
    """
    Linearly interpolate the values in `arr` at the locations `x`, in bins.
    For locations `x` outside the original array, extrapolate the function
    periodically.
    """
    n = arr.shape[-1]
    if not hasattr(x, 'shape'):
        x = np.array(x)
    floor = np.floor(x)
    t = x - floor
    pre_idx = floor.astype(np.int64) % n
    post_idx = np.ceil(x).astype(np.int64) % n
    pre_val = arr.take(pre_idx)
    post_val = arr.take(post_idx)
    interp_val = (1-t)*pre_val + t*post_val

    return interp_val[()]

def offpulse_window(profile, size):
    '''
    Find the off-pulse window of a given profile, defined as the
    segment of pulse phase of length `size` (in phase bins)
    minimizing the integral of the pulse profile.
    '''
    bins = np.arange(len(profile))
    lower = np.argmin(rolling_sum(profile, size))
    upper = lower + size
    return np.logical_and(lower <= bins, bins < upper)

def offpulse_rms(profile, size):
    '''
    Calculate the off-pulse RMS of a profile (a measure of noise level).
    This is the RMS of `profile` in the segment of length `size`
    (in phase bins) minimizing the integral of `profile`.
    '''
    opw = offpulse_window(profile, size)
    return np.sqrt(np.mean(profile[opw]**2))

def rolling_sum(arr, size):
    '''
    Calculate the sum of values in `arr` in a sliding window of length `size`,
    wrapping around at the end of the array.
    '''
    n = len(arr)
    s = np.cumsum(arr)
    return np.array([s[(i+size)%n]-s[i]+(i+size)//n*s[-1] for i in range(n)])

def symmetrize_limits(data, vmin=None, vmax=None):
    '''
    Produce symmetric limits for a set of data based on the data itself and
    (optionally) explicitly supplied upper or lower limits.
    '''
    datamin, datamax = np.nanmin(data), np.nanmax(data)
    lim = max(-datamin, datamax)
    if vmax is not None:
        lim = min(lim, vmax)
    if vmin is not None:
        lim = min(lim, -vmin)
    vmin, vmax = -lim, lim
    return vmin, vmax
