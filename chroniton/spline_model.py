import numpy as np
from scipy.interpolate import splev
from scipy.signal import resample
import pickle

from .portrait import Portrait

class SplineModel:
    def __init__(self, mean_prof, eigvec, tck):
        """
        Create a new spline model of a pulse portrait.

        Parameters
        ----------
        mean_prof: Mean profile shape. Should have shape (nbin,).
        eigvec: Array of eigenvectors, of shape (nbin, neig).
        tck: Tuple containing knot locations, B-spline coefficients,
             and spline degree (as output by `scipy.interpolate.splprep()`).
        """
        self.mean_prof = mean_prof
        self.eigvec = eigvec
        self.tck = tck

    @classmethod
    def from_file(cls, filename):
        """
        Read a spline model from a pickle file like that output by
        PulsePortraiture.

        Parameters
        ----------
        filename: Path to pickle file from which to read the spline model.
        """
        with open(filename, "rb") as f:
            model = pickle.load(f, encoding='latin1')
        modelname, source, datafile, mean_prof, eigvec, tck = model
        return cls(mean_prof, eigvec, tck)

    def make_portrait(self, freqs, nbin=None):
        """
        Generate a portrait based on this spline model.
        Equivalent to PulsePortraiture's `pplib.gen_spline_portrait()`.

        Parameters
        ----------
        freqs: Frequencies (in MHz) at which to evaluate the model.
        nbin: Number of phase bins to use in the model.
              Data will be resampled if necessary.
        """
        if self.eigvec.shape[1] > 0:
            proj_port = np.array(splev(freqs, self.tck, der=0, ext=0)).T
            delta_port = np.dot(proj_port, self.eigvec.T)
            port = delta_port + self.mean_prof
        else:
            port = np.broadcast_to(self.mean_prof, (freqs.size, self.mean_prof.size))
            port = port.copy()
        if nbin is not None and (nbin != self.mean_prof.shape[-1]):
            shift = 0.5 * (nbin**-1 - len(mean_prof)**-1)
            port = scipy.signal.resample(port, nbin, axis=1)
            port = rotate_portrait(port, shift) #resample introduces shift!
        return Portrait(freqs, port)
