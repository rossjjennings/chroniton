import numpy as np
from scipy.optimize import minimize_scalar
from collections import namedtuple

from .utils import fft_roll, offpulse_window, offpulse_rms, rolling_sum

ToaResult = namedtuple('ToaResult', ['toa', 'error', 'ampl'])

def toa_fourier(template, profile, ts = None, noise_level = None, tol = np.sqrt(np.finfo(np.float64).eps)):
    '''
    Calculate a TOA by maximizing the CCF of the template and the profile
    in the frequency domain. Searches within the interval between the sample
    below and the sample above the argmax of the circular CCF.

    `ts`:  Evenly-spaced array of phase values corresponding to the profile.
           Sets the units of the TOA. If this is `None`, the TOA is reported
           in bins.
    `tol`: Relative tolerance for optimization (in bins).
    `noise_level`: Off-pulse noise, in the same units as the profile.
           Used in calculating error. If not supplied, noise level will be
           estimated as the standard deviation of the profile residual.
    '''
    n = len(profile)
    if ts is None:
        ts = np.arange(n)
    dt = float(ts[1] - ts[0])

    template_fft = np.fft.fft(template)
    profile_fft = np.fft.fft(profile)
    phase_per_bin = -2j*np.pi*np.fft.fftfreq(n)

    circular_ccf = np.fft.irfft(np.fft.rfft(profile)*np.conj(np.fft.rfft(template)), n)
    ccf_argmax = np.argmax(circular_ccf)
    if ccf_argmax > n/2:
        ccf_argmax -= n
    ccf_max = ccf_argmax*dt

    def ccf_fourier(tau):
        phase = phase_per_bin*tau/dt
        ccf = np.inner(profile_fft, np.exp(-phase)*np.conj(template_fft))/n
        return ccf.real

    brack = (ccf_max - dt, ccf_max, ccf_max + dt)
    toa = minimize_scalar(lambda tau: -ccf_fourier(tau),
                          method = 'Brent', bracket = brack, tol = tol*dt).x

    assert brack[0] < toa < brack[-1]

    template_shifted = fft_roll(template, toa/dt)
    b = np.dot(template_shifted, profile)/np.dot(template, template)
    residual = profile - b*template_shifted
    ampl = b*np.max(template_shifted)
    if noise_level is None:
        noise_level = offpulse_rms(profile, profile.size//4)
    snr = ampl/noise_level

    w_eff = np.sqrt(n*dt/np.trapz(np.gradient(template, ts)**2, ts))
    error = w_eff/(snr*np.sqrt(n))

    return ToaResult(toa=toa, error=error, ampl=ampl)

def make_toas(template, portrait):
    toas = []
    errs = []
    ampls = []
    for port_profile in portrait:
        pass
        result = toa_fourier(template.I, port_profile)
        toas.append(result.toa)
        errs.append(result.error)
        ampls.append(result.ampl)
    toas = np.array(toas)
    errs = np.array(errs)
    ampls = np.array(ampls)
    return toas, errs, ampls
