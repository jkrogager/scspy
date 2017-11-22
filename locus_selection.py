import numpy as np
import locus_tools as lt
from os import path

root_path = path.dirname(path.abspath(__file__))


def run_locus_selection(colors, errors, N_err=4, locus='ugri'):
    """
    Test whether input colors are inside the stellar locus defined
    in either ``ugri`` or ``griz`` color-spaces.

    Parameters
    ----------
    colors : array_like, shape (N, 3)
        An array of N color vectors consisting of 3 colors:
        For ``ugri``: u-g, g-r, r-i ; and for ``griz``: g-r, r-i, i-z.

    errors : array_like, shape (N, 3)
        An array of N vectors containing the error on the input `colors`.

    N_err : int   [default = 4]
        The number of sigma used for the convolution of the stellar locus
        with the error ellipsoid defined by `errors`.

    locus : str   [default = 'ugri']
        Which stellar locus to work on, must be either ``ugri`` or ``griz``.

    Returns
    -------
    in_locus : array(bool), shape (N)
        Boolean array of length N for each target in the input `colors`.
        The array will be `True` if the target is inside the stellar locus
        convolved by `N_err` times the `errors`.
    """

    colors = np.array(colors)
    errors = np.array(errors)

    # Check input data format:
    assert colors.shape == errors.shape, "Input data and errors must have same dimensions."

    if len(colors.shape) == 2:
        if colors.shape[1] == 3:
            pass
        else:
            err_msg = "Input only allows three colours. Not %r given."
            raise ValueError(err_msg % colors.shape[1])
    else:
        err_msg = "Wrong input dimensions: %r"
        raise ValueError(err_msg % colors.shape)

    # --- Load ugri Stellar Locus:
    # i     k       N      u-g     g-r     r-i    k_ug    k_gr    k_ri      a       b      theta
    if locus.lower() == 'ugri':
        locus_pars = np.loadtxt(root_path+'/ugri_table_York2002.dat')
        # these are set manually for ugri and griz:
        a_k = 0.2
        k_end = -0.05

    elif locus.lower() == 'griz':
        locus_pars = np.loadtxt(root_path+'/griz_table_York2002.dat')
        # these are set manually for ugri and griz:
        a_k = 0.5
        k_end = -0.3
    O = locus_pars[:, 3:6]

    in_locus = list()
    for c, e in zip(colors, errors):
        loc_i = lt.find_nearest_locus_point(c, O)

        # --- Set locus parameters:
        k = locus_pars[loc_i, 6:9]
        orig = locus_pars[loc_i, 3:6]
        a_l = locus_pars[loc_i, 9]
        a_m = locus_pars[loc_i, 10]
        theta = locus_pars[loc_i, 11]

        # Generate 3D error ellipsoid:
        var = e**2
        cov = np.identity(3)*var
        cov_4sig = cov * N_err**2
        inv_cov = np.linalg.inv(cov_4sig)
        # Project data point to locus-plane:
        A_ij = lt.project_ellipsoid_to_plane(inv_cov, k)

        S_ij = lt.generate_inv_covariance_matrix(a_l, a_m, theta)

        # convolve A_ij and S_ij:
        C_ij = np.linalg.inv(A_ij) + np.linalg.inv(S_ij)
        # get new eigen values and eigen vectors
        eigvals, eigvecs = np.linalg.eig(C_ij)
        # update a_l and a_m with the new convolved values a_l' and a_m':
        a_lV = np.sqrt(max(eigvals))
        a_mV = np.sqrt(min(eigvals))
        l_prime = eigvecs[np.argmax(eigvals)]
        # update theta to theta':
        thetaV = np.arctan2(l_prime[1], l_prime[0])

        # convert covariance to inverse covariance:
        C_ij = np.linalg.inv(C_ij)

        # project observed colors to locus-plane:
        p_ij = lt.project_point_to_plane(c, k, orig)

        # Check if p_ij is in C_ij:
        in_ellipse_ij = lt.point_in_ellipse(p_ij, C_ij)
        if (c-orig).dot(k) > k_end:
            in_ellipse_ij *= True
        else:
            in_ellipse_ij = False

        if loc_i == 0:
            # --- Check if point is in end ellipsoid:

            # center of end ellipsoid:
            center_end = orig + k_end*k

            # define basis for ellipsoid:
            j = np.cross(k, lt.z)
            j = j/np.linalg.norm(j)
            i = np.cross(j, k)
            # The end ellipsoid is now defined in l' and m' space:
            l = np.cos(thetaV)*i + np.sin(thetaV)*j
            m = np.cos(thetaV + np.pi/2.)*i + np.sin(thetaV + np.pi/2.)*j

            # basis matrix for ellipsoid:
            U = np.column_stack([l, m, k])

            # extend a_k with projected variance:
            V_k = lt.project_ellipsoid_to_line(inv_cov, k)
            a_kV = np.sqrt(a_k**2 + V_k)
            # matrix for ellipsoid:
            S_end = np.array([[1./a_lV, 0., 0.],
                              [0., 1./a_mV, 0.],
                              [0., 0., 1./a_kV]])
            # inverse covariance matrix:
            C_xyz = U.dot((S_end**2).dot(U.T))

            in_ellipse_xyz = lt.point_in_ellipse(c, C_xyz, center_end)

            in_locus.append(bool(in_ellipse_ij + in_ellipse_xyz))
        else:
            in_locus.append(in_ellipse_ij)

    return np.array(in_locus, dtype=bool)
