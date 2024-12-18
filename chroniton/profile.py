import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal
from astropy.io import fits

from .utils import fft_roll
from .polarization import validate_stokes, coherence_to_stokes

class Profile:
    def __init__(self, I, Q=None, U=None, V=None):
        """
        Create a new profile from I, Q, U, and V arrays.
        If one of Q, U, or V is present, all must be, and all must have the same shape as I.
        """
        self.set_data(I, Q, U, V)

    def set_data(self, I, Q=None, U=None, V=None):
        """
        Set I, Q, U, and V, making sure the shapes match and dependent parameters
        (full_stokes, shape, nbin) are set accordingly.
        """
        self.full_stokes, self.shape = validate_stokes(I, Q, U, V)
        self.nbin = self.shape[-1]
        self.I = I
        if self.full_stokes:
            self.Q = Q
            self.U = U
            self.V = V

    @property
    def phase(self):
        """
        Array of phase values.
        """
        return np.linspace(0, 1, self.nbin, endpoint=False)

    @property
    def squared_norm(self):
        """
        The squared invariant interval (I**2 - Q**2 - U**2 - V**2).
        """
        if self.full_stokes:
            return self.I**2 - self.Q**2 - self.U**2 - self.V**2
        else:
            return self.I**2

    @classmethod
    def from_file(cls, filename):
        """
        Create a new template profile from a PSRFITS file.
        The data in the file should be fully scrunched in frequency and time
        (i.e., one subintegration, one channel).
        """
        hdul = fits.open(filename)
        data = hdul['SUBINT'].data['DATA']

        if data.shape[0] != 1:
            raise ValueError(f"Template should have 1 subintegration (found {data.shape[0]}).")
        if data.shape[2] != 1:
            raise ValueError(f"Template should have 1 channel (found {data.shape[2]}).")

        npol = data.shape[1]
        newshape = (1, npol, 1, 1)
        scale = hdul['SUBINT'].data['DAT_SCL'].reshape(newshape)
        offset = hdul['SUBINT'].data['DAT_OFFS'].reshape(newshape)
        profile = data*scale + offset

        pol_type = hdul['SUBINT'].header['POL_TYPE'].upper()
        feed_poln = hdul['PRIMARY'].header['FD_POLN'].upper()
        hdul.close()
        if pol_type in ['AA+BB', 'INTEN']:
            # Total intensity data
            I = profile.flatten()
            return cls(I)
        elif pol_type == 'IQUV':
            # Full Stokes data
            I, Q, U, V = profile.reshape(4, -1)
            return cls(I, Q, U, V)
        elif pol_type == 'AABBCRCI':
            # Coherence data - convert to Stokes
            AA, BB, CR, CI = profile.reshape(4, -1)
            I, Q, U, V = coherence_to_stokes(AA, BB, CR, CI, feed_poln)
            return cls(I, Q, U, V)
        else:
            raise ValueError(f"Unrecognized polarization type '{pol_type}'.")

    def normalize(self):
        """
        Normalize to a maximum amplitude of unity.
        """
        I_max = np.max(self.I)
        self.I /= I_max
        if self.full_stokes:
            self.Q /= I_max
            self.U /= I_max
            self.V /= I_max

    def make_posdef(self, fudge_factor=1.5):
        """
        Add a small constant to the total intensity to make sure the
        squared invariant interval is positive.

        Parameters
        ----------
        fudge_factor: Overcompensate by this factor, so as to avoid
                      having the invariant interval equal to zero at
                      the phase where it is minimized.
        """
        adjustment = max(-np.min(self.squared_norm), 0.0)
        adjustment *= fudge_factor
        print(f"Adjusting I**2 by {adjustment}")
        self.I = np.sqrt(self.I**2 + adjustment)

    def plot(self, ax=None, what='IQUV', colors=None, shift=0.0, **kwargs):
        """
        Plot the profile.

        Parameters
        ----------
        ax: Axes on which to plot template. If `None`, a new Figure and
            Axes will be created.
        what: Which polarization to plot: 'I', 'ILV', or 'IQUV'.
              Ignored if template only has total intensity data.
        shift: Rotation (in cycles) to apply before plotting.
        colors: Dictionary mapping polarization component names
                ('I', 'L', 'Q', 'U', 'V') to colors to use for plotting.
                If `None`, use a default set of colors.

        Additional keyword arguments are passed on to ax.plot().
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        if colors is None:
            colors = {
                'I': 'k',
                'L': 'C3',
                'Q': 'C1',
                'U': 'C2',
                'V': 'C0',
                'S': 'C4',
                'S$^2$': 'C4',
            }

        plot_arrays = {'I': self.I}
        if what == 'S':
            plot_arrays = {'S': np.sqrt(self.squared_norm)}
        elif what == 'S2':
            plot_arrays = {'S$^2$': self.squared_norm}
        elif self.full_stokes and what == 'ILV':
            plot_arrays['L'] = np.sqrt(self.Q**2 + self.U**2)
            plot_arrays['V'] = self.V
        elif self.full_stokes and what == 'IQUV':
            plot_arrays['Q'] = self.Q
            plot_arrays['U'] = self.U
            plot_arrays['V'] = self.V

        artists = []
        for name, arr in plot_arrays.items():
            phase = self.phase - shift
            arr_shifted = fft_roll(arr, shift*self.nbin)
            lines = ax.plot(phase, arr_shifted, label=name,
                            color=colors[name], **kwargs)
            artists.extend(lines)

        ax.set_xlabel("Phase (cycles)")
        ax.set_ylabel("Intensity")
        ax.legend()
        return artists

    def resample(self, nbin):
        """
        Resample the profile to a given number of phase bins.
        Returns a new Profile object.
        """
        I = scipy.signal.resample(self.I, nbin)
        if self.full_stokes:
            Q = scipy.signal.resample(self.Q, nbin)
            U = scipy.signal.resample(self.U, nbin)
            V = scipy.signal.resample(self.V, nbin)
            return Profile(I, Q, U, V)
        else:
            return Profile(I)
