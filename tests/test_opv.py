from dataclasses import replace
import unittest
import numpy as np
from oled_opv_lab.opv import OPVParameters,calculate_opv,load_solar_spectrum,photon_flux,HC_J_NM,integrate


class OPVTests(unittest.TestCase):
    def test_reference_spectrum_and_photon_conversion(self):
        w,i=load_solar_spectrum()
        self.assertEqual(len(w),2002)
        self.assertAlmostEqual(float(w[0]),280)
        self.assertAlmostEqual(float(w[-1]),4000)
        self.assertAlmostEqual(integrate(i,w),1000.3706555734423,places=5)
        self.assertFalse(w.flags.writeable)
        np.testing.assert_allclose(photon_flux([500,600],[1,2]),np.array([500,1200])/HC_J_NM)

    def test_alignment_and_offset_tradeoff(self):
        a=calculate_opv(OPVParameters(offset_ev=.1))
        b=calculate_opv(OPVParameters(offset_ev=.6))
        self.assertEqual(a['absorbed_photons_m2_s'],b['absorbed_photons_m2_s'])
        self.assertGreater(b['separation_yield'],a['separation_yield'])
        self.assertGreater(b['jsc_ma_cm2'],a['jsc_ma_cm2'])
        self.assertLess(b['voc_v'],a['voc_v'])
        self.assertAlmostEqual(a['donor_lumo_ev']-a['acceptor_lumo_ev'],.1)
        self.assertAlmostEqual(a['ct_energy_ev'],a['acceptor_lumo_ev']-a['donor_homo_ev'])
        zero=calculate_opv(OPVParameters(offset_ev=0))
        self.assertEqual(zero['jsc_a_m2'],0)
        self.assertEqual(zero['voc_v'],0)
        self.assertEqual(zero['power_w_m2'],0)
        mid=calculate_opv(OPVParameters(offset_ev=.3))
        end=calculate_opv(OPVParameters(offset_ev=1))
        self.assertGreater(mid['power_w_m2'],end['power_w_m2'])

    def test_intensity_and_unit_conversions(self):
        w,i=load_solar_spectrum()
        a=calculate_opv()
        b=calculate_opv(spectrum=(w,2*i))
        self.assertAlmostEqual(b['absorbed_photons_m2_s']/a['absorbed_photons_m2_s'],2)
        self.assertAlmostEqual(b['jsc_a_m2']/a['jsc_a_m2'],2)
        self.assertAlmostEqual(b['power_w_m2']/a['power_w_m2'],2)
        self.assertAlmostEqual(a['jsc_ma_cm2'],a['jsc_a_m2']*.1)
        self.assertAlmostEqual(a['power_w_m2'],a['jsc_a_m2']*a['voc_v']*.7)

    def test_bounds_and_no_free_power(self):
        for gap in np.linspace(1,3.5,8):
            for offset in np.linspace(0,1,8):
                r=calculate_opv(OPVParameters(gap_ev=float(gap),offset_ev=float(offset)))
                self.assertTrue(np.all((r['absorptance']>=0)&(r['absorptance']<=.85)))
                self.assertGreaterEqual(r['power_w_m2'],0)
                self.assertLessEqual(r['power_w_m2'],r['absorbed_power_w_m2'])
        self.assertEqual(calculate_opv(OPVParameters(absorptance_max=0))['power_w_m2'],0)
        self.assertEqual(calculate_opv(OPVParameters(fill_factor=0))['power_w_m2'],0)
        self.assertEqual(calculate_opv(OPVParameters(gap_ev=1,offset_ev=1))['voc_v'],0)
        with self.assertRaises(ValueError):calculate_opv(OPVParameters(offset_ev=-.1))
        with self.assertRaises(ValueError):calculate_opv(OPVParameters(gap_ev=float('nan')))
