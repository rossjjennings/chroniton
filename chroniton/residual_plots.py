import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u
import os.path
from ruamel.yaml import YAML
yaml = YAML(typ='safe')

with open(os.path.join(os.path.dirname(__file__), 'colorschemes.yaml')) as f:
    colorschemes = yaml.load(f)

def plot_residuals(resids, x='time', y=None, yerr=None, cat=None, ax=None, grid=True, legend=True,
                   whiten=False, colorscheme='ipta', colorby='pta', avg=False, marker=None, label=None):
    standalone = False
    if ax is None:
        fig = plt.figure(figsize=(9.6, 4.8))
        ax = fig.add_subplot()
        standalone = True

    if x == 'time':
        xlabel = "MJD"
    elif x == 'freq':
        xlabel = "Frequency (MHz)"
    elif x == 'orbphase':
        xlabel = "Orbital phase (cycles)"

    if cat is None:
        cat = np.array(resids.toas[colorby]) # absent flag = ''
        cat[cat == ''] = 'None'

    x, y, yerr, cat = get_plot_values(resids, x, y, yerr, cat, whiten=whiten, avg=avg)
    colorscheme = colorschemes[colorscheme]
    artists = plot_categorized(x, y, yerr, cat, ax, colorscheme[colorby], marker=marker, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Residual (\N{MICRO SIGN}s)")
    if grid:
        ax.grid()
    if legend:
        ax.legend(loc='lower center', bbox_to_anchor= (0.5, 1.0), ncol=5)
    if standalone:
        plt.tight_layout()
    return artists

def get_plot_values(resids, x='time', y=None, yerr=None, cat=None, whiten=False, avg=False):
    xtype = x
    if x == 'freq':
        x = resids.toas.get_freqs()
    elif x == 'orbphase':
        if resids.model.BINARY.value is None:
            raise ValueError("Model has no binary component, cannot get orbital phase")
        x = resids.model.orbital_phase(resids.toas)
    elif x == 'time':
        x = resids.toas.get_mjds()
    if y is None:
        y = resids.time_resids.copy()
        if whiten:
            print('Whitening...')
            if not resids.noise_resids:
                warnings.warn('')
            if 'pl_red_noise' in resids.noise_resids:
                print('Removing red noise')
                y -= resids.noise_resids['pl_red_noise']
            if 'pl_DM_noise' in resids.noise_resids:
                print('Removing DM noise')
                y -= resids.noise_resids['pl_DM_noise']
        #if correct_offset:
        #    y -= np.dot(fitter.current_state.M[:,0], fitter.current_state.xhat[0]) * u.s
        #if correct_timing_model:
        #fitp = fitter.model.get_params_dict("free", "quantity")
        #    ntmpar = len(fitp)
        #    y -= np.dot(fitter.current_state.M[:,1:1+ntmpar], fitter.current_state.xhat[1:1+ntmpar]) * u.s
    if yerr is None:
        yerr = resids.get_data_error()

    if avg:
        if xtype == 'time':
            groupid = resids.toas['name']
            x_unit = u.d
        elif xtype == 'freq':
            fs = resids.toas['f']
            chans = resids.toas['chan']
            groupid = np.array([f"{f}.{chan:>02}" for (f, chan) in zip(fs, chans)])
            x_unit = u.MHz
        elif xtype == 'orbphase':
            groupid = resids.toas['name']
            x_unit = u.Unit()
        uniq, idx, inv = np.unique(groupid, return_index=True, return_inverse=True)
        x = np.bincount(inv, weights=x.to(x_unit).value)/np.bincount(inv)*x_unit
        y = np.bincount(inv, weights=np.float64(y.to(u.us).value))/np.bincount(inv)*u.us
        yerr = np.sqrt(np.bincount(inv, weights=yerr.to(u.us).value**2)/np.bincount(inv)**2)*u.us
        if cat is not None:
            cat = cat[idx]

    if cat is not None:
        return x, y, yerr, cat
    else:
        return x, y, yerr

def plot_categorized(x, y, yerr, cat, ax, colors, marker=None, label=None):
    artists = []
    for category in np.unique(cat):
        mask = (cat == category)
        color, default_marker = colors[category]
        if marker is None:
            marker = default_marker
        if label is None:
            cat_label = category
        else:
            cat_label = f"{category} ({label})"
        eb = ax.errorbar(
            x[mask].value,
            y[mask].to(u.us),
            yerr[mask].to(u.us),
            ls='',
            color=color,
            marker=marker,
            alpha=0.5,
            label=cat_label,
        )
        artists.extend(eb.get_children())
    return artists
