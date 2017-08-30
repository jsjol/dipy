import numpy as np
from scipy.linalg import cho_factor, cho_solve

def dki_design_matrix(gtab):
    r""" Constructs B design matrix for DKI

    Parameters
    ---------
    gtab : GradientTable
        Measurement directions.

    Returns
    -------
    B : array (N, 22)
        Design matrix or B matrix for the DKI model
        B[j, :] = (Bxx, Bxy, Bzz, Bxz, Byz, Bzz,
                   Bxxxx, Byyyy, Bzzzz, Bxxxy, Bxxxz,
                   Bxyyy, Byyyz, Bxzzz, Byzzz, Bxxyy,
                   Bxxzz, Byyzz, Bxxyz, Bxyyz, Bxyzz,
                   BlogS0)
    """
    b = gtab.bvals
    bvec = gtab.bvecs

    B = np.zeros((len(b), 22))
    B[:, 0] = -b * bvec[:, 0] * bvec[:, 0]
    B[:, 1] = -2 * b * bvec[:, 0] * bvec[:, 1]
    B[:, 2] = -b * bvec[:, 1] * bvec[:, 1]
    B[:, 3] = -2 * b * bvec[:, 0] * bvec[:, 2]
    B[:, 4] = -2 * b * bvec[:, 1] * bvec[:, 2]
    B[:, 5] = -b * bvec[:, 2] * bvec[:, 2]
    B[:, 6] = b * b * bvec[:, 0]**4 / 6
    B[:, 7] = b * b * bvec[:, 1]**4 / 6
    B[:, 8] = b * b * bvec[:, 2]**4 / 6
    B[:, 9] = 4 * b * b * bvec[:, 0]**3 * bvec[:, 1] / 6
    B[:, 10] = 4 * b * b * bvec[:, 0]**3 * bvec[:, 2] / 6
    B[:, 11] = 4 * b * b * bvec[:, 1]**3 * bvec[:, 0] / 6
    B[:, 12] = 4 * b * b * bvec[:, 1]**3 * bvec[:, 2] / 6
    B[:, 13] = 4 * b * b * bvec[:, 2]**3 * bvec[:, 0] / 6
    B[:, 14] = 4 * b * b * bvec[:, 2]**3 * bvec[:, 1] / 6
    B[:, 15] = b * b * bvec[:, 0]**2 * bvec[:, 1]**2
    B[:, 16] = b * b * bvec[:, 0]**2 * bvec[:, 2]**2
    B[:, 17] = b * b * bvec[:, 1]**2 * bvec[:, 2]**2
    B[:, 18] = 2 * b * b * bvec[:, 0]**2 * bvec[:, 1] * bvec[:, 2]
    B[:, 19] = 2 * b * b * bvec[:, 1]**2 * bvec[:, 0] * bvec[:, 2]
    B[:, 20] = 2 * b * b * bvec[:, 2]**2 * bvec[:, 0] * bvec[:, 1]
    B[:, 21] = np.ones(len(b))

    return B

def probabilistic_least_squares(design_matrix, y, regularization_matrix=None, posterior_samples=None):
    # Solve least-squares problem on the form
    # design_matrix * coef = y

    if regularization_matrix is None:
        unscaled_posterior_precision = np.dot(design_matrix.T, design_matrix)
    else:
        unscaled_posterior_precision = np.dot(design_matrix.T, design_matrix) + regularization_matrix

    pseudoInv = np.linalg.solve(unscaled_posterior_precision, design_matrix.T)
    coef_posterior_mean = np.dot(pseudoInv, y)

    smoother_matrix = design_matrix.dot(pseudoInv)
    residual_matrix = np.eye(y.shape[0]) - smoother_matrix
    residual_variance = (np.linalg.norm(residual_matrix.dot(y)) ** 2 /
                         np.linalg.norm(residual_matrix, 'fro') ** 2)

    if posterior_samples is None:
        return coef_posterior_mean, residual_variance
    else:
        standard_normal_samples = np.random.randn(coef_posterior_mean.shape[0], posterior_samples)

        coef_posterior_precision = unscaled_posterior_precision / residual_variance
        L = cho_factor(coef_posterior_precision)

        if np.ndim(coef_posterior_mean) == 1:
            # For correct broadcasting
            coef_posterior_mean = coef_posterior_mean[:, None]

        samples = coef_posterior_mean + cho_solve(L, standard_normal_samples)

        samples = np.squeeze(samples)

        return samples, residual_variance
