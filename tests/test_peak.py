import unittest
import numpy as np
from oled_opv_lab.solvation import emission_peak, gaussian_energy, wavelength_spectrum, sigma_to_fwhm, HC


class PeakTests(unittest.TestCase):
    def test_peak_matches_maximum_on_both_spectral_axes(self):
        centre,sigma=2.4,.25
        width=sigma_to_fwhm(sigma)
        energy,wavelength=emission_peak(centre,sigma)
        self.assertAlmostEqual(energy,centre)
        self.assertAlmostEqual(energy*wavelength,HC)
        grid=np.linspace(energy-.01,energy+.01,1001)
        self.assertAlmostEqual(float(grid[np.argmax(gaussian_energy(grid,centre,width))]),energy)
        energy,wavelength=emission_peak(centre,sigma,wavelength_axis=True)
        self.assertAlmostEqual(energy*wavelength,HC)
        grid=np.linspace(wavelength-1,wavelength+1,1001)
        self.assertAlmostEqual(float(grid[np.argmax(wavelength_spectrum(grid,centre,width))]),wavelength)
        self.assertNotAlmostEqual(wavelength,HC/centre)
