"""
ATHERTON: Phase Coded Orientation Annulus circle detection.

Implements the convolution-based circle detector from:
    Atherton, T.J. & Kerbyson, D.J. (1999). Size invariant circle detection.
    Image and Vision Computing, 17(11), 795-803.

The method convolves image gradients with a complex-valued Phase Coded
Orientation Annulus (PCOA) kernel.  The peak magnitude in the output gives
the circle centre; the phase at the peak encodes the radius via log-phase
coding (Eq. 8 of the paper).

Main function: atherton(img, R_min=5, R_max=20)
"""

import numpy as np
from scipy.signal import fftconvolve


def _build_pcoa_kernels(R_min, R_max):
    """
    Build the Phase Coded Orientation Annulus convolution kernels.

    Returns two complex-valued kernels K_x and K_y of shape
    (2*R_max+1, 2*R_max+1).

    For each position (m, n) inside the annulus R_min <= r <= R_max:
        O_OAx(m,n) = cos(theta)          orientation annulus x-component (Eq. 4)
        O_OAy(m,n) = sin(theta)          orientation annulus y-component
        O_PCA(m,n) = exp(i * phi)         phase coded annulus            (Eq. 5)
        phi        = 2*pi * (log(r) - log(R_min)) / (log(R_max) - log(R_min))
                                          log phase coding               (Eq. 8)
        K_x = (1/r) * O_OAx * O_PCA      combined kernel, x-gradient    (Eq.16)
        K_y = (1/r) * O_OAy * O_PCA      combined kernel, y-gradient

    Parameters
    ----------
    R_min : int
        Minimum detectable radius.
    R_max : int
        Maximum detectable radius.

    Returns
    -------
    K_x, K_y : numpy.ndarray (complex128)
        Convolution kernels for x- and y-gradient channels.
    """
    size = 2 * R_max + 1
    K_x = np.zeros((size, size), dtype=np.complex128)
    K_y = np.zeros((size, size), dtype=np.complex128)

    log_Rmin = np.log(max(R_min, 0.5))
    log_Rmax = np.log(R_max)
    log_range = log_Rmax - log_Rmin
    if log_range == 0:
        log_range = 1.0  # degenerate case

    center = R_max  # kernel centre index

    for m in range(-R_max, R_max + 1):
        for n in range(-R_max, R_max + 1):
            r = np.sqrt(m * m + n * n)
            if r < R_min or r > R_max:
                continue

            theta = np.arctan2(n, m)                        # Eq. 4
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            phi = 2.0 * np.pi * (np.log(r) - log_Rmin) / log_range  # Eq. 8
            pca = np.exp(1j * phi)                          # Eq. 5

            inv_r = 1.0 / r                                 # Eq. 9-11
            K_x[center + m, center + n] = inv_r * cos_t * pca
            K_y[center + m, center + n] = inv_r * sin_t * pca

    return K_x, K_y


def _decode_radius(phase, R_min, R_max):
    """
    Recover the detected radius from the phase at the accumulator peak.

    phase = 2*pi * (log(R) - log(R_min)) / (log(R_max) - log(R_min))
    => R = exp(log(R_min) + phase/(2*pi) * (log(R_max) - log(R_min)))

    Parameters
    ----------
    phase : float
        Phase angle in radians (may be negative; wrapped to [0, 2*pi)).
    R_min, R_max : int
        Radius limits used when building the kernel.

    Returns
    -------
    R : float
        Decoded radius in pixels.
    """
    phase_wrapped = phase % (2.0 * np.pi)
    log_Rmin = np.log(max(R_min, 0.5))
    log_Rmax = np.log(R_max)
    R = np.exp(log_Rmin + (phase_wrapped / (2.0 * np.pi)) * (log_Rmax - log_Rmin))
    return R


def atherton(img, R_min=5, R_max=20):
    """
    Detect a circle using the Phase Coded Orientation Annulus method.

    Parameters
    ----------
    img : numpy.ndarray
        Input grayscale image (uint8 or float).  The method computes
        Sobel gradients internally.
    R_min : int, optional
        Minimum circle radius to detect (default: 5).
    R_max : int, optional
        Maximum circle radius to detect (default: 20).

    Returns
    -------
    x, y, r : float
        Circle centre (x=col, y=row) and radius.
        Returns (0, 0, 0) if no circle is detected.

    Notes
    -----
    The convolution kernel size is (2*R_max+1) x (2*R_max+1).  For R_max=20
    this is 41x41, which is fast even on small ROI images.

    The method pads the image so that circles near the border can still be
    detected.
    """
    if img is None or img.size == 0:
        return 0.0, 0.0, 0.0

    # Convert to float
    img_f = img.astype(np.float64)

    # Compute Sobel gradients  (E_x, E_y in the paper)
    # Using 3x3 Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64) / 8.0
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float64) / 8.0

    E_x = fftconvolve(img_f, sobel_x, mode='same')
    E_y = fftconvolve(img_f, sobel_y, mode='same')

    # Build PCOA kernels
    K_x, K_y = _build_pcoa_kernels(R_min, R_max)

    # Convolve: Q_PCOA = E_x (*) K_x  +  E_y (*) K_y      (Eq. 16)
    # Flip kernels for correlation→convolution equivalence
    Q = fftconvolve(E_x, K_x[::-1, ::-1], mode='same') + \
        fftconvolve(E_y, K_y[::-1, ::-1], mode='same')

    # Magnitude map — peak = circle centre
    mag = np.abs(Q)

    if mag.max() == 0:
        return 0.0, 0.0, 0.0

    peak_idx = np.unravel_index(np.argmax(mag), mag.shape)
    y_c = float(peak_idx[0])   # row
    x_c = float(peak_idx[1])   # col

    # Decode radius from phase at the peak
    phase_at_peak = np.angle(Q[peak_idx])
    r_c = _decode_radius(phase_at_peak, R_min, R_max)

    # Sanity check
    if r_c < R_min * 0.5 or r_c > R_max * 1.5:
        return 0.0, 0.0, 0.0

    return x_c, y_c, r_c
