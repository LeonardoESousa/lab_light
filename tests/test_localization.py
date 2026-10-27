import unittest
import numpy as np
from oled_opv_lab.solvation import (localization_properties, CHI_MAX_EV,
    emission_energy, emission_sigma, sigma_to_fwhm, gaussian_energy, wavelength_spectrum)
from oled_opv_lab.colours import scale_emission_color


class LocalizationTests(unittest.TestCase):
    def test_limits_symmetry_and_monotonicity(self):
        for angle in (0,30,45,90):
            p=localization_properties(angle,angle)
            self.assertEqual(p['chi'],0)
            self.assertEqual(p['strength'],1)
        p=localization_properties(0,90)
        self.assertEqual(p['chi'],CHI_MAX_EV)
        self.assertEqual(p['strength'],0)
        middle=localization_properties(0,45)
        self.assertAlmostEqual(middle['overlap'],.5)
        self.assertEqual(localization_properties(20,70),localization_properties(70,20))
        grid=[localization_properties(0,a) for a in range(91)]
        self.assertTrue(np.all(np.diff([p['chi'] for p in grid])>=0))
        self.assertTrue(np.all(np.diff([p['strength'] for p in grid])<=0))
        with self.assertRaises(ValueError):localization_properties(-1,45)

    def test_intensity_area_and_dark_limit(self):
        e=np.linspace(.01,8,30000)
        w=np.linspace(200,2500,30000)
        for angles in [(0,0),(0,45),(0,90)]:
            p=localization_properties(*angles)
            peak=emission_energy(2.9,p['chi'],.6,.3)
            width=sigma_to_fwhm(emission_sigma(p['chi'],.6,.3))
            y=p['strength']*gaussian_energy(e,peak,width)
            ynm=p['strength']*wavelength_spectrum(w,peak,width)
            integrate=getattr(np,'trapezoid',None) or np.trapz
            self.assertAlmostEqual(float(integrate(y,e)),p['strength'],places=6)
            self.assertAlmostEqual(float(integrate(ynm,w)),p['strength'],places=5)
        self.assertEqual(scale_emission_color((0,255,128),0),(0,0,0))
        self.assertEqual(scale_emission_color((0,255,128),1),(0,255,128))
        self.assertLess(scale_emission_color((0,255,128),.25)[1],255)
