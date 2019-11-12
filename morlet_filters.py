import numpy as np


def compute_sigma_psi(xi, Q, r=np.sqrt(0.5)):
    """
    Computes the frequency width of a morlet wavelet from its center
    frequency, under constant-Q assumption.

    Inputs:
    -------
    xi : float
        center frequency, between -0.5 and 0.5
    Q : int
        number of wavelet per octave/scale
    r : float (optional)
        overlap parameter used in the design of the wavelet
    
    Output:
    -------
    sigma : float
        frequency standard deviation of morlet wavelet
    """
    factor = 1. / np.power(2, 1. / Q)
    term1 = (1 - factor) / (1 + factor)
    term2 = 1. / np.sqrt(2 * np.log(1. / r))
    sigma = xi * term1 * term2
    return sigma


def periodize_filter_fourier(filt, nperiods):
    """
    Computes the aliased version of filter in Fourier domain,
    which corresponds to subsampling in time.

    Inputs:
    -------
    filt : numpy array of shape (T*nperiods,)
        fourier filter to be periodized
    nperiods : odd int
        number of periods

    Output :
    -------
    periodic_filt : numpy array of shape (T,)
        Aliased version of filt
    """

    true_size = np.size(filt) // nperiods
    if nperiods%2 == 0:
        filt = np.roll(filt, true_size // 2)

    filt = np.split(filt, nperiods)
    filt = np.sum(filt, axis=0)
    return filt


def morlet_1d(N, xi, Q, P=5, eps=1e-7):
    """
    Computes the Fourier transform of a Morlet filter.
    A Morlet filter is the sum of a Gabor filter and a low-pass filter
    to ensure that the sum has exactly zero mean in the temporal domain.
    It is defined by the following formula in time:
    psi(t) = g_{sigma}(t) (e^{i xi t} - beta)
    where g_{sigma} is a Gaussian envelope, xi is a frequency and beta is
    the cancelling parameter.
    Parameters
    ----------
    N : int
        size of the temporal support
    xi : float
        central frequency (in [0, 1])
    Q : int
        bandwidth parameter, number of wavelets per octave
    normalize : string, optional
        normalization types for the filters. Defaults to 'l1'.
        Supported normalizations are 'l1' and 'l2' (understood in time domain).
    P_max: int, optional
        integer controlling the maximal number of periods to use to ensure
        the periodicity of the Fourier transform. (At most 2*P_max - 1 periods
        are used, to ensure an equal distribution around 0.5). Defaults to 5
        Should be >= 1
    eps : float
        required machine precision (to choose the adequate P)
    Returns
    -------
    morlet_f : array_like
        numpy array of size (N,) containing the Fourier transform of the Morlet
        filter at the frequencies given by np.fft.fftfreq(N).
    """
    assert isinstance(P, int)

    # get sigma from xi with constant Q basis
    sigma = compute_sigma_psi(xi, Q)
    
    freqs = np.arange((1 - P) * N, P * N, dtype=float) / float(N)
    
    # define gabor envelope and low pass in Fourier domain
    gabor_f = np.exp(-(freqs - xi)**2 / (2 * sigma**2))
    low_pass_f = np.exp(-(freqs**2) / (2 * sigma**2))
    
    # Discretize <=> periodize in Fourier (Aliasing)
    gabor_f = periodize_filter_fourier(gabor_f, 2*P-1)
    low_pass_f = periodize_filter_fourier(low_pass_f, 2*P-1)

    # get low_pass coefficient such that wavelet has zero mean
    kappa = gabor_f[0] / low_pass_f[0]
    morlet_f = gabor_f - kappa * low_pass_f
    
    morlet_f = morlet_f / np.max(np.abs(morlet_f))

    return morlet_f
