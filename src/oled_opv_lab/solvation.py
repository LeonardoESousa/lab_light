"""Phenomenological solvatochromism; all energy parameters are in eV."""
from dataclasses import dataclass
import numpy as np

HC = 1239.8419843320026  # eV nm
KB_EV = 8.617333262145e-5  # eV K^-1
TEMPERATURE_K = 300.0


@dataclass(frozen=True)
class Molecule:
    name: str
    e_vac: float
    chi: float


MOLECULES = (
    Molecule('A · weakly sensitive emitter', 2.9, 0.12),
    Molecule('B · moderately sensitive emitter', 2.9, 0.65),
    Molecule('C · strongly sensitive emitter', 2.9, 1.25),
    Molecule('D · lower-energy emitter', 2.4, 0.55),
    Molecule('E · negative solvatochromism', 2.4, -0.55),
)


def response(dielectric):
    if not np.isfinite(dielectric) or dielectric < 1:
        raise ValueError('Dielectric constant must be finite and at least 1.')
    return (dielectric-1)/(dielectric+1)


def emission_energy(e_vac, chi, alpha_st, alpha_opt):
    if not np.all(np.isfinite([e_vac, chi, alpha_st, alpha_opt])):
        raise ValueError('All parameters must be finite.')
    if not 0 <= alpha_opt <= alpha_st < 1:
        raise ValueError('Require 0 ≤ αopt ≤ αst < 1.')
    energy = e_vac-chi*(2*alpha_st-alpha_opt)
    if energy <= 0:
        raise ValueError('This parameter combination gives a nonpositive transition energy.')
    return float(energy)


def emission_sigma(chi, alpha_st, alpha_opt, temperature=TEMPERATURE_K):
    """User's phenomenological Gaussian sigma (eV), not FWHM.

    A signed solvatochromic chi need not be a nonnegative reorganization
    parameter. Reject nonpositive variance rather than taking abs or clipping.
    """
    if not np.all(np.isfinite([chi, alpha_st, alpha_opt, temperature])):
        raise ValueError('Linewidth parameters must be finite.')
    if not 0 <= alpha_opt <= alpha_st < 1 or temperature <= 0:
        raise ValueError('Require 0 ≤ αopt ≤ αst < 1 and positive temperature.')
    kt = KB_EV * temperature
    variance = kt**2 + 2*chi*(alpha_st-alpha_opt)*kt
    if variance <= 0:
        raise ValueError('The linewidth formula gives nonpositive σ². A negative solvatochromic χ cannot generally also serve as a nonnegative broadening parameter. Choose a positive χ or a smaller static–optical response difference.')
    return float(np.sqrt(variance))


def sigma_to_fwhm(sigma):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Sigma must be positive and finite.')
    return float(2*np.sqrt(2*np.log(2))*sigma)


def gaussian_energy(energy, peak, fwhm):
    if not np.isfinite(fwhm) or fwhm <= 0 or not np.isfinite(peak) or peak <= 0:
        raise ValueError('Peak energy and FWHM must be positive and finite.')
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    return np.exp(-.5*((np.asarray(energy)-peak)/sigma)**2)/(sigma*np.sqrt(2*np.pi))


def wavelength_spectrum(wavelength_nm, peak, fwhm):
    """Energy-density Gaussian transformed with |dE/dλ|; not a Gaussian in nm."""
    w = np.asarray(wavelength_nm, dtype=float)
    if np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError('Wavelengths must be finite and positive.')
    return gaussian_energy(HC/w, peak, fwhm)*HC/w**2
