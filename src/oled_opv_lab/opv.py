"""Photon harvesting and charge-generation/voltage trade-off teaching model.

The Gaussian is dimensionless absorptance, not a normalized spectral density.
Power is a score of this phenomenological model, not a calibrated device PCE.
"""
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import numpy as np

Q = 1.602176634e-19  # C, also J/eV
HC_J_NM = 6.62607015e-34 * 299792458 * 1e9
HC_EV_NM = HC_J_NM / Q
DONOR_HOMO_EV = -5.5
ACCEPTOR_HOMO_EV = -6.2  # spectator level; only donor absorption is included
ABSORPTANCE_MAX = .85
FWHM_EV = .45
SEPARATION_SCALE_EV = .15
LOSS_EV = .30
FILL_FACTOR = .70


def integrate(y, x):
    function = getattr(np, 'trapezoid', None) or np.trapz
    return float(function(y, x))


@lru_cache(maxsize=1)
def load_solar_spectrum():
    """ASTM G173 global-tilt data in nm and W m^-2 nm^-1, bundled offline."""
    with files('oled_opv_lab').joinpath('data/am15g.csv').open('r') as handle:
        data = np.loadtxt(handle, delimiter=',', skiprows=1)
    wavelength, irradiance = data[:, 0].copy(), data[:, 1].copy()
    if (len(wavelength)<2 or np.any(~np.isfinite(data)) or np.any(np.diff(wavelength)<=0)
            or np.any(wavelength<=0) or np.any(irradiance<0)):
        raise ValueError('The bundled reference solar spectrum is invalid.')
    wavelength.setflags(write=False)
    irradiance.setflags(write=False)
    return wavelength, irradiance


def photon_flux(wavelength_nm, irradiance):
    w, intensity = np.asarray(wavelength_nm, float), np.asarray(irradiance, float)
    if (w.ndim!=1 or intensity.shape!=w.shape or len(w)<2 or np.any(~np.isfinite(w))
            or np.any(w<=0) or np.any(np.diff(w)<=0)
            or np.any(~np.isfinite(intensity)) or np.any(intensity<0)):
        raise ValueError('Supply matching, ordered positive wavelengths and nonnegative irradiance.')
    return intensity * w / HC_J_NM


@dataclass(frozen=True)
class OPVParameters:
    gap_ev: float = 1.9
    offset_ev: float = .3
    fwhm_ev: float = FWHM_EV
    absorptance_max: float = ABSORPTANCE_MAX
    separation_scale_ev: float = SEPARATION_SCALE_EV
    loss_ev: float = LOSS_EV
    fill_factor: float = FILL_FACTOR

    def validate(self):
        values = [self.gap_ev, self.offset_ev, self.fwhm_ev, self.absorptance_max,
                  self.separation_scale_ev, self.loss_ev, self.fill_factor]
        if not np.all(np.isfinite(values)):
            raise ValueError('All OPV parameters must be finite.')
        if (self.gap_ev<=0 or self.offset_ev<0 or self.fwhm_ev<=0
                or self.separation_scale_ev<=0 or self.loss_ev<0
                or not 0<=self.absorptance_max<=1 or not 0<=self.fill_factor<=1):
            raise ValueError('Invalid OPV gap, offset, linewidth, loss, absorptance or fill factor.')


def calculate_opv(parameters=OPVParameters(), spectrum=None):
    parameters.validate()
    w, irradiance = load_solar_spectrum() if spectrum is None else spectrum
    w, irradiance = np.asarray(w, float), np.asarray(irradiance, float)
    flux = photon_flux(w, irradiance)
    p = parameters
    energies = HC_EV_NM/w
    sigma = p.fwhm_ev/(2*np.sqrt(2*np.log(2)))
    absorptance = p.absorptance_max*np.exp(-.5*((energies-p.gap_ev)/sigma)**2)
    absorbed_flux = flux*absorptance
    absorbed_photons = integrate(absorbed_flux,w)
    incident_photons = integrate(flux,w)
    separation_yield = float(-np.expm1(-p.offset_ev/p.separation_scale_ev)) if p.offset_ev>0 else 0.
    collected_flux = separation_yield*absorbed_flux
    jsc = Q*absorbed_photons*separation_yield
    ct_energy = p.gap_ev-p.offset_ev
    voltage_ceiling = max(ct_energy-p.loss_ev,0.)
    # No photogenerated current means no photovoltage in this teaching limit.
    voc = voltage_ceiling if jsc>0 else 0.
    power = jsc*voc*p.fill_factor
    return dict(wavelength_nm=w,irradiance_w_m2_nm=irradiance,photon_flux_nm=flux,
                absorptance=absorptance,absorbed_flux_nm=absorbed_flux,collected_flux_nm=collected_flux,
                absorbed_photons_m2_s=absorbed_photons,incident_photons_m2_s=incident_photons,
                absorbed_fraction=absorbed_photons/incident_photons if incident_photons>0 else 0.,
                absorbed_power_w_m2=integrate(irradiance*absorptance,w),
                incident_power_w_m2=integrate(irradiance,w),separation_yield=separation_yield,
                jsc_a_m2=jsc,jsc_ma_cm2=.1*jsc,voc_v=voc,ct_energy_ev=ct_energy,
                power_w_m2=power,donor_homo_ev=DONOR_HOMO_EV,
                donor_lumo_ev=DONOR_HOMO_EV+p.gap_ev,acceptor_homo_ev=ACCEPTOR_HOMO_EV,
                acceptor_lumo_ev=DONOR_HOMO_EV+p.gap_ev-p.offset_ev)
