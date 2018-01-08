# -*- coding: UTF-8 -*-
import numpy as np
from locus_selection import run_locus_selection

"""
    Calculate the detection probability in given bins of z_abs, NHI, Av parameter space
    for various quasar redshifts.

    call with argument EXT_LAW:
    > python zAv_full_selection2.py  EXT_LAW
"""


def get_color_vector(mag, err):
    """Return the 4-color vector from 5-point photometry"""
    colors = np.array([mag[i] - mag[i+1] for i in range(len(mag)-1)])
    errors = np.array([np.sqrt(err[i]**2 + err[i+1]**2) for i in range(len(mag)-1)])
    return colors, errors


# Parameters from noise locus fitting
# One row corresponds to one filter (u,g,r,i,z)
P_noise = np.array([[-1.282e-03, 1.0097e-01, -2.933, 3.7432e+01, -1.793e+02],
                    [-1.079e-03, 8.9617e-02, -2.736, 3.6587e+01, -1.830e+02],
                    [-7.049e-04, 6.0503e-02, -1.880, 2.5347e+01, -1.276e+02],
                    [-9.142e-04, 7.5581e-02, -2.276, 2.9835e+01, -1.461e+02],
                    [-7.840e-04, 5.6493e-02, -1.472, 1.6558e+01, -6.978e+01]])


def noise_function(mag):
    """
    Calculate one sigma uncertainty on magnitudes from polynomial fit to mag-sigma distribution.
    The distribution is fitted in terms of mag vs. log10(sigma).
    """
    sigma = np.zeros_like(mag)
    sigma_lower = np.zeros(5)
    sigma_upper = np.zeros(5)
    order = 4
    for i in range(order+1):
        sigma += P_noise.T[i] * mag**(order-i)

    for j in range(5):
        sigma_lower[j] = np.poly1d(P_noise[j])(14.0)
        sigma_upper[j] = np.poly1d(P_noise[j])(22.5)

        # Check for outliers:
        high = mag[:, j] > 22.5
        low = mag[:, j] < 14.0
        sigma[:, j][high] = sigma_upper[j]
        sigma[:, j][low] = sigma_lower[j]

    return 10**sigma


def color_selection(sample, sample_error, verbose=True):
    """
    Run full SDSS quasar candidates selection as specified in
    Richards et al. (2002, AJ 123, 2945-2975).
    All the color and photometric criteria are implemented.
    The returned arrays contain `True` if the given set of
    photometry has passed the criteria, and `False` otherwise.

    Parameters
    ----------

    sample : array_like, shape (N, 5)
        Input photometry in five SDSS bands: u, g, r, i, z
        The array should contain a column for each filter.

    sample_error : array_like, shape (N, 5)
        Input 1-sigma uncertainty for photometry in five bands.
        Should be same dimensions as `sample`.

    verbose : bool   [default = True]
        If `True`, print status messages.

    Returns
    -------

    output : dict
        Dictionary containing the following keys:

        'QSO_FULL' :
            Boolean array of full combined ugri and griz selection.
            Identical to `QSO_UGRI * QSO_UGRI_PHOT + QSO_GRIZ * QSO_GRIZ_PHOT`.

        'QSO_COLOR' :
            Boolean array of pure `ugri` + `griz` color selection, i.e., neglecting the
            i-band criteria for `ugri` and `griz`.

        'QSO_PHOT' :
            Boolean array of pure `ugri` + `griz` photometric selection,
            i.e., only i < 20.2. Identical to `QSO_GRIZ_PHOT`.

        'QSO_GRIZ' :
            Boolean array of full `griz` color selection.

        'QSO_UGRI' :
            Boolean array of full `ugri` color selection.

        'QSO_GRIZ_PHOT' :
            Boolean array of griz i<20.2 criterion only.

        'QSO_UGRI_PHOT' :
            Boolean array of ugri i<19.1 criterion only.

        'QSO_GRIZ_COLOR' :
            Boolean array of pure griz color criterion only.

        'QSO_UGRI_COLOR' :
            Boolean array of pure ugri color criterion only.

        'REJECT' :
            Boolean array of targets fulfilling the rejection criteria.
            These are made up by white dwarf, A-star and red-blue pair
            exclusion regions. See Richards et al. 2002.

    """

    sample = np.array(sample)
    sample_error = np.array(sample_error)

    colors, errors = get_color_vector(sample.T, sample_error.T)

    # For ugri, use first 3 colors:
    ugri_points = colors.T[:, :3]
    ugri_errors = errors.T[:, :3]
    # For griz, use last 3 colors:
    griz_points = colors.T[:, 1:]
    griz_errors = errors.T[:, 1:]

    # Unpack separate bands and errors:
    u_mag = sample[:, 0]
    g_mag = sample[:, 1]
    r_mag = sample[:, 2]
    i_mag = sample[:, 3]
    z_mag = sample[:, 4]

    u_err = sample_error[:, 0]
    g_err = sample_error[:, 1]
    r_err = sample_error[:, 2]
    i_err = sample_error[:, 3]
    z_err = sample_error[:, 4]

    # Container for target rejection:
    # targets start out as TRUE and if they fail a criterion they are turned to FAIL
    N_targets = len(u_mag)
    is_quasar = np.ones(N_targets, dtype=bool)

    if verbose:
        print("\n  Running SDSS Color Selection Algorithm")
        print("  Implemented in Python by J.-K. Krogager")
        print("  Reference: Richards et al. 2002, AJ 123, 2945\n")
        if N_targets == 1:
            print("  Running on %i target" % N_targets)
        else:
            print("  Running on %i targets" % N_targets)

    # ==========================================================================
    # --- EXCLUSION REGIONS:

    # griz exclusion region:
    griz_ex = g_mag - r_mag < 1.0
    griz_ex *= (u_mag - g_mag >= 0.8)
    griz_ex *= (i_mag >= 19.1) + (u_mag - g_mag < 2.5)

    # white dwarf exclusion region:
    WD_ex = (u_mag - g_mag > -0.8) * (u_mag - g_mag < 0.7)
    WD_ex *= (g_mag - r_mag > -0.8) * (g_mag - r_mag < -0.1)
    WD_ex *= (r_mag - i_mag > -0.6) * (r_mag - i_mag < -0.1)
    WD_ex *= (i_mag - z_mag > -1.0) * (i_mag - z_mag < -0.1)

    # A star exclusion region:
    A_ex = (u_mag - g_mag > 0.7) * (u_mag - g_mag < 1.4)
    A_ex *= (g_mag - r_mag > -0.5) * (g_mag - r_mag < 0.0)
    A_ex *= (r_mag - i_mag > -0.5) * (r_mag - i_mag < 0.2)
    A_ex *= (i_mag - z_mag > -0.4) * (i_mag - z_mag < 0.2)

    # WD+M pair exclusion region:
    WDM_ex = (g_mag - r_mag > -0.3) * (g_mag - r_mag < 1.25)
    WDM_ex *= (r_mag - i_mag > 0.6) * (r_mag - i_mag < 2.0)
    WDM_ex *= (i_mag - z_mag > 0.4) * (i_mag - z_mag < 1.2)
    WDM_ex *= g_err < 0.2

    # Test if photometry is in exclusion region:
    reject = WD_ex + A_ex + WDM_ex
    is_quasar = is_quasar * ~reject
    # ==========================================================================

    # ==========================================================================
    #     UGRI SELECTION:
    ugri_cand = is_quasar.copy()

    # not in ugri stellar locus (4 sigma)
    in_ugri = run_locus_selection(ugri_points[~reject], ugri_errors[~reject],
                                  N_err=4, locus='ugri')
    in_ugri = np.array(in_ugri, dtype=bool)
    ugri_cand[~reject] = ~in_ugri

    # or in UVX box:
    UVX = (u_err < 0.1) * (g_err < 0.1)
    UVX *= u_mag - g_mag < 0.6
    ugri_cand = ugri_cand + UVX*~reject

    # or in mid-z region:
    # 2.5 < z < 3 inclusion, 2-sigma locus:
    midz_in = (u_mag - g_mag > 0.6) * (u_mag - g_mag < 1.5)
    midz_in *= (g_mag - r_mag > 0.0) * (g_mag - r_mag < 0.2)
    midz_in *= (r_mag - i_mag > -0.1) * (r_mag - i_mag < 0.4)
    midz_in *= (i_mag - z_mag > -0.1) * (i_mag - z_mag < 0.4)
    midz_in *= ~reject

    in_2sig_ugri = run_locus_selection(ugri_points[midz_in], ugri_errors[midz_in],
                                       N_err=2, locus='ugri')
    in_2sig_ugri = np.array(in_2sig_ugri, dtype=bool)
    midz_selected = ~in_2sig_ugri

    # Select only 10% of the objects in this region:
    midz_qso = midz_selected == 1
    if np.sum(midz_qso) > 10:
        qso_subset = midz_selected[midz_qso]
        random10 = np.random.choice(len(qso_subset), len(qso_subset)/10, replace=False)
        qso_subset[:] = 0
        qso_subset[random10] = 1
        midz_selected[midz_qso] = qso_subset
        ugri_cand[midz_in] = midz_selected

    # magnitude criteria 15 < i < 19.1:
    ugri_mag_cut = (i_mag > 15.0) * (i_mag < 19.1)
    ugri_cand_magcut = ugri_cand * ugri_mag_cut

    # ==========================================================================
    #     GRIZ SELECTION:
    griz_cand = is_quasar.copy()

    # not in griz stellar locus (4 sigma)
    in_griz = run_locus_selection(griz_points[~reject], griz_errors[~reject],
                                  N_err=4, locus='griz')
    in_griz = np.array(in_griz, dtype=bool)
    griz_cand[~reject] = ~in_griz

    # reject low-z interlopers:
    lowz_rej = g_mag - r_mag < 1.0
    lowz_rej *= u_mag - g_mag >= 0.8
    lowz_rej *= (i_mag >= 19.1) + (u_mag - g_mag < 2.5)
    griz_cand = griz_cand*~lowz_rej

    # or in gri inclusion for z>3.6; (6) of Richards et al. 2002
    gri_in = i_err < 0.2
    gri_in *= (u_mag - g_mag > 1.5) + (u_mag > 20.6)
    gri_in *= g_mag - r_mag > 0.7
    gri_in *= (g_mag - r_mag > 2.1) + (r_mag - i_mag < 0.44*(g_mag - r_mag) - 0.358)
    gri_in *= i_mag - z_mag < 0.25
    gri_in *= i_mag - z_mag > -1.0
    griz_cand = griz_cand + gri_in*~reject

    # riz inclusion for z>4.5; (7) of Richards et al. 2002
    riz_in = i_err < 0.2
    riz_in *= u_mag > 21.5
    riz_in *= g_mag > 21.0
    riz_in *= r_mag - i_mag > 0.6
    riz_in *= i_mag - z_mag > -1.0
    riz_in *= (i_mag - z_mag < 0.52*(r_mag - i_mag) - 0.412)
    griz_cand = griz_cand + riz_in*~reject

    # ugr red outliers for z>3.0; (8) of Richards et al. 2002
    ugr_red1 = u_mag > 20.6
    ugr_red1 *= u_mag - g_mag > 1.5
    ugr_red1 *= g_mag - r_mag < 1.2
    ugr_red1 *= r_mag - i_mag < 0.3
    ugr_red1 *= i_mag - z_mag > -1.0
    ugr_red1 *= (g_mag - r_mag < 0.44*(u_mag - g_mag) - 0.56)

    # ugri outliers from the stellar locus can be selected in griz if:
    ugr_red2 = is_quasar.copy()
    ugr_red2[~reject] = ~in_ugri
    ugr_red2 *= u_err < 0.2
    ugr_red2 *= g_err < 0.2
    ugr_red2 *= u_mag - g_mag > 1.5

    ugr_red = ugr_red1 + ugr_red2
    griz_cand = griz_cand + ugr_red*~reject

    # magnitude criteria 15 < i < 20.2:
    griz_mag_cut = (i_mag < 20.2) * (i_mag > 15.0)
    griz_cand_magcut = griz_cand * griz_mag_cut
    # ==========================================================================

    # Combine candidates from ugri and griz selections:
    is_quasar_col = ugri_cand + griz_cand
    is_quasar_phot = ugri_mag_cut + griz_mag_cut
    is_quasar_full = ugri_cand_magcut + griz_cand_magcut

    # Pack output:
    output = dict()
    output['QSO_FULL'] = is_quasar_full
    output['QSO_COLOR'] = is_quasar_col
    output['QSO_PHOT'] = is_quasar_phot
    output['QSO_GRIZ'] = griz_cand_magcut
    output['QSO_UGRI'] = ugri_cand_magcut
    output['QSO_GRIZ_COLOR'] = griz_cand
    output['QSO_UGRI_COLOR'] = ugri_cand
    output['QSO_GRIZ_PHOT'] = griz_mag_cut
    output['QSO_UGRI_PHOT'] = ugri_mag_cut
    output['REJECT'] = reject

    if verbose:
        N_qso = np.sum(is_quasar_full)
        if N_qso == 1:
            print("\n  Identified %i target as quasar candidates." % N_qso)
        else:
            print("\n  Identified %i targets as quasar candidates." % N_qso)

    return output
