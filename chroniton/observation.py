import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal
from astropy.io import fits
import astropy.units as u
from astropy.time import Time
from pint import PulsarMJD

from .utils import fft_roll
from .polarization import validate_stokes, coherence_to_stokes
from .portrait import Portrait

class Observation:
    def __init__(self, epochs, freq, I, Q=None, U=None, V=None):
        """
        Create a new observation from I, Q, U, and V arrays.
        If one of Q, U, or V is present, all must be, and all must have the same shape as I.
        """
        self.epochs = epochs
        self.freq = freq

        self.full_stokes, self.shape = validate_stokes(I, Q, U, V)
        self.I = I
        if self.full_stokes:
            self.Q = Q
            self.U = U
            self.V = V

        self.nbin = self.shape[-1]
        self.phase = np.linspace(0, 1, self.nbin, endpoint=False)

    @classmethod
    def from_file(cls, filename):
        """
        Create a new observation from a PSRFITS file.
        """
        hdul = fits.open(filename)
        data = hdul['SUBINT'].data['DATA']
        dat_scl = hdul['SUBINT'].data['DAT_SCL']
        dat_offs = hdul['SUBINT'].data['DAT_OFFS']
        dat_freq = hdul['SUBINT'].data['DAT_FREQ']
        start_mjd = hdul['PRIMARY'].header['STT_IMJD']
        start_sec = hdul['PRIMARY'].header['STT_SMJD']
        start_offs = hdul['PRIMARY'].header['STT_OFFS']
        offs_sub = hdul['SUBINT'].data['OFFS_SUB']
        pol_type = hdul['SUBINT'].header['POL_TYPE'].upper()
        feed_poln = hdul['PRIMARY'].header['FD_POLN'].upper()

        nsub, npol, nchan, nbin = data.shape
        newshape = (nsub, npol, nchan, 1)
        scale = dat_scl.reshape(newshape)
        offset = dat_offs.reshape(newshape)
        data = data*scale + offset

        freq = dat_freq[0]*u.MHz
        start_time = Time(start_mjd, format='pulsar_mjd')
        start_time += start_sec*u.s
        start_time += start_offs*u.s
        epochs = start_time + offs_sub*u.s
        hdul.close()

        if pol_type in ['AA+BB', 'INTEN']:
            # Total intensity data
            I, = data.transpose(1, 0, 2, 3)
            return cls(epochs, freq, I)
        elif pol_type == 'IQUV':
            # Full Stokes data
            I, Q, U, V = data.transpose(1, 0, 2, 3)
            return cls(epochs, freq, I, Q, U, V)
        elif pol_type == 'AABBCRCI':
            # Coherence data - convert to Stokes
            AA, BB, CR, CI = data.transpose(1, 0, 2, 3)
            I, Q, U, V = coherence_to_stokes(AA, BB, CR, CI, feed_poln)
            return cls(epochs, freq, I, Q, U, V)
        else:
            raise ValueError(f"Unrecognized polarization type '{pol_type}'.")

    def avg_portrait(self, noise_weight=True, unit_max=False):
        I = np.nanmean(self.I, axis=0)
        if self.full_stokes:
            Q = np.nanmean(self.Q, axis=0)
            U = np.nanmean(self.U, axis=0)
            V = np.nanmean(self.V, axis=0)
            return Portrait(self.freq, I, Q, U, V)
        else:
            return Portrait(self.freq, I)

    def __getitem__(self, key):
        I = self.I[key, ...]
        if self.full_stokes:
            Q = self.Q[key, ...]
            U = self.U[key, ...]
            V = self.V[key, ...]
            return Portrait(self.freq, I, Q, U, V)
        else:
            return Portrait(self.freq, I)
