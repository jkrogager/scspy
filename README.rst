
==============
SCSPy
==============

.. role:: red

.. include:: <isonum.txt>

This module contains the code to run the quasar color selection algorithm
used for the Sloan Digital Sky Survey before the launch of SDSS-III (BOSS).
The algorithm has been reimplemented in Python using linear algebra for the
stellar locus rejection. The original algorithm is presented by `Richards et al.
(2002) <https://ui.adsabs.harvard.edu/#abs/2002AJ....123.2945R/abstract>`_.

The algorithm takes standard SDSS photometry as input,
i.e., magnitudes in 5 bands (u, g, r, i, z) and their corresponding errors.
These should be passed as an array-like structure each with a shape (N, 5),
for N targets.

Running the code from a python script only requires an import of the module
and then the function `color_selection()` can be run as follows:

.. code-block:: python

    import scspy
    qso_cand = scspy.color_selection(photometry, errors, verbose=False)


The output variable `qso_cand` is a dictionary containing arrays of
shape (N) for N targets where each array gives details of the color
selection tags for each target. The dictionary keys are as follows:

        'QSO_FULL' :
            Boolean array of full combined ugri and griz selection.
            Identical to `QSO_UGRI * QSO_UGRI_PHOT + QSO_GRIZ * QSO_GRIZ_PHOT`.

        'QSO_COLOR' :
            Boolean array of pure `ugri` + `griz` color selection, i.e., neglecting the
            i-band criteria for `ugri` and `griz`.

        'QSO_GRIZ' :
            Boolean array of pure `griz` color selection.

        'QSO_UGRI' :
            Boolean array of pure `ugri` color selection.

        'QSO_GRIZ_PHOT' :
            Boolean array of griz i<20.2 criterion only.

        'QSO_UGRI_PHOT' :
            Boolean array of ugri i<19.1 criterion only.

        'REJECT' :
            Boolean array of targets fulfilling the rejection criteria.
            These are made up by white dwarf, A-star and red-blue pair
            exclusion regions. See Richards et al. 2002.

|

.. important::
    :class: red

    The `griz` selection branch currently suffers from some uncertainty
    due to an error in the low-redshift rejection criteria as stated in
    equation 1 of `Richards et al. (2002)
    <https://ui.adsabs.harvard.edu/#abs/2002AJ....123.2945R/abstract>`_.
    The algorithm has been tested and optimized using a dataset of 10,000
    quasars from SDSS DR7 and recovers roughly the right number of `QSO_HIZ'
    targets from the `griz` branch. However, we caution that the criteria
    implemented in the low-redshift rejection algorithm as part of the `griz`
    selection are reverse-engineered in order to obtain a number of
    `QSO_HIZ' targets that matches as closely as possible the number of targets
    in the test data.
