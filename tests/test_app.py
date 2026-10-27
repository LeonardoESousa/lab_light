import json
import base64
import unittest
import numpy as np
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP=Path(__file__).parents[1]/'src'/'oled_opv_lab'/'app.py'


def chart_spec(at, title):
    for chart in at.get('plotly_chart'):
        spec=json.loads(chart.proto.spec)
        if spec['layout']['title']['text']==title:
            return spec
    raise AssertionError(f'Chart not found: {title}')


def values(data):
    if isinstance(data,dict):
        return np.frombuffer(base64.b64decode(data['bdata']),dtype=data['dtype'])
    return np.asarray(data)


class DashboardTests(unittest.TestCase):
    def test_both_activities_react_and_attempts_save(self):
        at=AppTest.from_file(str(APP),default_timeout=30)
        for i,value in enumerate((35.,15.,70.,45.)):
            at.session_state[f'angle_{i}']=value
        at.run()
        self.assertFalse(at.exception)
        before=at.metric[0].value
        at.slider(key='angle_1').set_value(40.).run()
        self.assertFalse(at.exception)
        self.assertNotEqual(at.metric[0].value,before)
        at.button(key='save_orbital').click().run()
        self.assertEqual(len(at.session_state['st_attempts']),1)
        before=next(c.value for c in at.caption if 'FWHM =' in c.value)
        at.slider(key='epsilon').set_value(12.).run()
        self.assertFalse(at.exception)
        self.assertNotEqual(next(c.value for c in at.caption if 'FWHM =' in c.value),before)
        at.slider(key='epsilon').set_value(1.).run()
        self.assertFalse(at.exception)
        self.assertTrue(any('Using εs' in i.value for i in at.info))
        at.number_input(key='ev').set_value(3.1).run()
        self.assertFalse(at.exception)
        at.radio(key='spectrum_axis').set_value('Energy / eV').run()
        self.assertFalse(at.exception)

    def test_random_initialization_persists_and_spacing_tracks_exchange(self):
        at=AppTest.from_file(str(APP),default_timeout=30).run()
        self.assertFalse(at.exception)
        original=[at.slider(key=f'angle_{i}').value for i in range(4)]
        self.assertTrue(all(0<a<90 for a in original))
        at.slider(key='angle_0').set_value(13.).run()
        self.assertEqual([at.slider(key=f'angle_{i}').value for i in range(4)],[13.,*original[1:]])
        at.slider(key='kmax').set_value(.8).run()
        self.assertFalse(at.exception)
        self.assertGreaterEqual(at.number_input(key='d_occ').value,1.65)
        self.assertGreaterEqual(at.number_input(key='d_vir').value,1.65)
        at.button(key='randomize_angles').click().run()
        self.assertFalse(at.exception)

    def test_peak_normalization_and_relative_intensity_view(self):
        at=AppTest.from_file(str(APP),default_timeout=30).run()
        self.assertFalse(at.exception)
        self.assertEqual(at.slider(key='n').value,1.4)
        self.assertEqual(at.slider(key='epsilon').value,2.)
        self.assertTrue(at.checkbox(key='normalize_spectra').value)
        initial=chart_spec(at,'Emission spectrum')
        for curve in initial['data']:
            self.assertAlmostEqual(float(values(curve['y']).max()),1.)
        at.checkbox(key='normalize_spectra').uncheck().run()
        relative=chart_spec(at,'Emission spectrum')
        maxima=[float(values(curve['y']).max()) for curve in relative['data']]
        self.assertAlmostEqual(relative['layout']['yaxis']['range'][1],1.08*max(maxima))
        self.assertNotEqual(maxima[0],maxima[1])
        at.checkbox(key='normalize_spectra').check().run()
        at.slider(key='solv_theta_h').set_value(0.).run()
        at.slider(key='solv_theta_l').set_value(90.).run()
        self.assertFalse(at.exception)
        dark=chart_spec(at,'Emission spectrum')
        self.assertEqual(initial['layout']['yaxis']['range'],dark['layout']['yaxis']['range'])
        for curve in dark['data']:
            self.assertTrue(np.all(values(curve['y'])==0))
        self.assertEqual(next(m.value for m in at.metric if m.label=='Relative emission strength'),'0.000')
        at.slider(key='solv_theta_l').set_value(0.).run()
        self.assertFalse(at.exception)
        self.assertEqual(next(m.value for m in at.metric if m.label=='Relative emission strength'),'1.000')


class OPVAppTests(unittest.TestCase):
    def test_opv_controls_and_saved_attempts(self):
        at=AppTest.from_file(str(APP),default_timeout=30).run()
        self.assertFalse(at.exception)
        def metric(label):return next(m.value for m in at.metric if m.label==label)
        photons=metric('Absorbed photons / m²/s')
        power=metric('Power score / W/m²')
        at.slider(key='opv_offset').set_value(0.).run()
        self.assertFalse(at.exception)
        self.assertEqual(metric('Absorbed photons / m²/s'),photons)
        self.assertEqual(metric('Power score / W/m²'),'0.00')
        at.slider(key='opv_offset').set_value(.3).run()
        self.assertEqual(metric('Power score / W/m²'),power)
        at.slider(key='opv_gap').set_value(2.5).run()
        self.assertFalse(at.exception)
        self.assertNotEqual(metric('Absorbed photons / m²/s'),photons)
        at.button(key='opv_save').click().run()
        self.assertEqual(len(at.session_state['opv_attempts']),1)
        self.assertEqual(at.session_state['opv_attempts'][0]['Donor peak / eV'],2.5)
        at.button(key='opv_clear').click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state['opv_attempts'],[])
