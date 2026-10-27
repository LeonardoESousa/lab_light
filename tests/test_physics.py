from dataclasses import replace
import numpy as np
import unittest
from oled_opv_lab import core
from oled_opv_lab.models import OrbitalParameters, solve, eligible, minimum_frontier_spacing
from oled_opv_lab.cisd_basis import SINGLET_SPACE, TRIPLET_SPACE
from oled_opv_lab.solvation import emission_energy, response, emission_sigma, sigma_to_fwhm, KB_EV, gaussian_energy, wavelength_spectrum, HC
from oled_opv_lab.colours import spectrum_to_color


class PhysicsTests(unittest.TestCase):

    def test_adda_state_switch_is_not_rewarded(self):
        # The old near-degenerate occupied pair allows H−1→L to undercut H→L.
        p = OrbitalParameters(angles=(90.,0.,0.,90.))
        switched = solve(p)
        self.assertFalse(eligible(switched))
        self.assertLess(switched['s1_hl'], .8)
        # The neighboring-orbital floor restores H→L in this regression case.
        d = minimum_frontier_spacing(p.k_max)
        isolated = solve(replace(p, energies=(0.,d,d+3.026,d+3.6019)))
        self.assertTrue(eligible(isolated))
        self.assertGreater(isolated['s1_hl'], .8)
        self.assertGreater(isolated['t1_hl'], .8)
        self.assertGreater(isolated['gap'], .1)

    def test_character_criterion_checks_both_states(self):
        r = dict(s1=3., t1=3., s1_hl=.99, t1_hl=.1)
        self.assertFalse(eligible(r))
        r.update(s1_hl=.1,t1_hl=.99)
        self.assertFalse(eligible(r))
        r.update(s1_hl=.8,t1_hl=.8)
        self.assertTrue(eligible(r))
        r['t1'] = -1.
        self.assertFalse(eligible(r))
    
    def test_spin_spaces_and_orthogonality(self):
        for space, size, spin in [(SINGLET_SPACE,15,0),(TRIPLET_SPACE,11,1)]:
            assert len(space.basis)==size
            matrix=np.array([[c.state.get(d,0) for c in space.basis] for d in core.MS0_DETERMINANTS])
            np.testing.assert_allclose(matrix.T@matrix,np.eye(size),atol=1e-12)
            for csf in space.basis:
                transformed=core.s2_apply_ms0(csf.state)
                np.testing.assert_allclose([transformed.get(d,0) for d in core.MS0_DETERMINANTS],
                                           [spin*(spin+1)*csf.state.get(d,0) for d in core.MS0_DETERMINANTS],atol=1e-12)
    
    
    def test_noninteracting_limit_and_zero_invariance(self):
        p=OrbitalParameters(k_max=0)
        r=solve(p)
        np.testing.assert_allclose(r['s1'],p.energies[2]-p.energies[1])
        np.testing.assert_allclose(r['t1'],r['s1'])
        np.testing.assert_allclose(r['gap'],0,atol=1e-12)
        q=solve(replace(p,energies=tuple(e+12 for e in p.energies)))
        np.testing.assert_allclose(q['s1'],r['s1'])
    
    
    def test_interacting_hamiltonians_and_coupling_limits(self):
        r=solve(OrbitalParameters(angles=(25.,10.,70.,45.)))
        np.testing.assert_allclose(r['hs'],r['hs'].T,atol=1e-12)
        np.testing.assert_allclose(r['ht'],r['ht'].T,atol=1e-12)
        assert 0<=r['s1_single']<=1+1e-12
        assert r['s2']>=r['s1']>0
        assert abs(r['gap'])>1e-5
        p=core.ModelParameters(0,np.pi/2,0,.5,3,4,.1)
        g=core.interaction_tensor(p)
        np.testing.assert_allclose(g[0,0,0,0],.1)
        np.testing.assert_allclose(g[1,2,2,1],0,atol=1e-15)
    
    
    def test_solvation_formula_and_validation(self):
        assert emission_energy(3,.5,0,0)==3
        np.testing.assert_allclose(emission_energy(3,.5,.4,.2),2.7)
        np.testing.assert_allclose(emission_energy(3,-.5,.4,.2),3.3)
        assert response(1)==0
        np.testing.assert_allclose(response(2),1/3)
        np.testing.assert_allclose(response(1.4**2),(1.4**2-1)/(1.4**2+1))
        # The corrected response factors can exceed the previous bound of 0.5.
        np.testing.assert_allclose(emission_energy(3,.5,.8,.6),2.5)
        with self.assertRaises(ValueError): emission_energy(3,.5,.1,.2)
        with self.assertRaises(ValueError): solve(OrbitalParameters(energies=(0,2,1,3)))

    def test_thermal_sigma_and_invalid_signed_chi(self):
        kt=KB_EV*300
        self.assertAlmostEqual(emission_sigma(0,.8,.2),kt)
        self.assertAlmostEqual(emission_sigma(1,.4,.4),kt)
        self.assertAlmostEqual(emission_sigma(.65,.6,.3),np.sqrt(kt**2+2*.65*.3*kt))
        self.assertGreater(emission_sigma(1.2,.6,.3),emission_sigma(.65,.6,.3))
        self.assertAlmostEqual(sigma_to_fwhm(kt),2*np.sqrt(2*np.log(2))*kt)
        with self.assertRaises(ValueError): emission_sigma(-.55,.8,.3)

    def test_vial_responds_to_broadening_at_fixed_peak(self):
        w=np.linspace(200,2500,4601)
        narrow=wavelength_spectrum(w,2.35,sigma_to_fwhm(emission_sigma(0,.7,.3)))
        broad=wavelength_spectrum(w,2.35,sigma_to_fwhm(emission_sigma(1.2,.7,.3)))
        self.assertNotEqual(spectrum_to_color(w,narrow),spectrum_to_color(w,broad))
    
    
    def test_spectral_jacobian_and_integral(self):
        energy=np.linspace(.05,7,50000)
        wavelength=HC/energy[::-1]
        a=gaussian_energy(energy,2.5,.3)
        b=wavelength_spectrum(wavelength,2.5,.3)
        np.testing.assert_allclose(np.trapz(a,energy),1,rtol=1e-6)
        np.testing.assert_allclose(np.trapz(b,wavelength),1,rtol=1e-6)
    
    
    def test_colour_is_finite_scale_invariant_and_not_white(self):
        w=np.linspace(250,2000,3501)
        y=wavelength_spectrum(w,2.7,.2)
        original=y.copy()
        c=spectrum_to_color(w,y)
        assert c==spectrum_to_color(w,y*10)
        assert max(c)==255 and min(c)<200
        np.testing.assert_array_equal(y,original)
        assert spectrum_to_color(w,np.zeros_like(w))==(24,28,38)
        assert spectrum_to_color(w,wavelength_spectrum(w,.7,.05))==(24,28,38)
