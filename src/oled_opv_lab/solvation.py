"""Phenomenological solvatochromism; all energy parameters are in eV."""
from dataclasses import dataclass
import numpy as np

HC = 1239.8419843320026  # eV nm
KB_EV = 8.617333262145e-5  # eV K^-1
TEMPERATURE_K = 300.0
CHI_MAX_EV = 1.25


def localization_properties(theta_homo, theta_lumo, chi_max=CHI_MAX_EV):
    """Map D/A localization to a similarity proxy, chi and relative strength.

    O=|u_H dot u_L|²=cos²(theta_H-theta_L), with u=(cos(theta),sin(theta)).
    This compares fragment descriptors, not actual orthogonal MO overlaps.
    The mappings chi=chi_max*(1-O), I_rel=O are pedagogical assumptions.
    """
    if not np.all(np.isfinite([theta_homo, theta_lumo, chi_max])):
        raise ValueError('Localization parameters must be finite.')
    if not (0 <= theta_homo <= 90 and 0 <= theta_lumo <= 90) or chi_max < 0:
        raise ValueError('Angles must be between 0 and 90 degrees; chi_max must be nonnegative.')
    overlap = float(np.clip(np.cos(np.radians(theta_homo-theta_lumo))**2, 0, 1))
    if overlap < 1e-14:
        overlap = 0.
    return dict(overlap=overlap, chi=chi_max*(1-overlap), strength=overlap)


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


def emission_peak(energy_centre, sigma, wavelength_axis=False):
    """Return eV/nm for the same maximum on the selected spectral-density axis.

    I_lambda = E²/(hc) I_E: its maximum obeys E²-mu*E-2*sigma²=0.
    Therefore simply relabeling hc/mu as a wavelength peak would be incorrect.
    """
    if not np.all(np.isfinite([energy_centre,sigma])) or energy_centre<=0 or sigma<=0:
        raise ValueError('Energy centre and sigma must be positive and finite.')
    e=float(energy_centre)
    if wavelength_axis:
        e=.5*(e+float(np.hypot(e,np.sqrt(8)*sigma)))
    return e, HC/e


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
