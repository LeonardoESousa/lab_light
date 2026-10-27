import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP=Path(__file__).parents[1]/'src'/'oled_opv_lab'/'app.py'


class DashboardTests(unittest.TestCase):
    
    def test_both_activities_react_and_attempts_save(self):
        at=AppTest.from_file(str(APP),default_timeout=30)
        for i, value in enumerate((35.,15.,70.,45.)):
            at.session_state[f'angle_{i}'] = value
        at.run()
        assert not at.exception
        before=at.metric[0].value
        at.slider(key='angle_1').set_value(40.).run()
        assert not at.exception
        assert at.metric[0].value!=before
        at.button(key='save_orbital').click().run()
        assert len(at.session_state['hl_attempts'])==1
        at.selectbox(key='molecule').select('Custom molecule').run()
        before=[m.value for m in at.metric]
        at.slider(key='chi').set_value(1.2).run()
        assert not at.exception
        assert [m.value for m in at.metric]!=before
        at.radio(key='solvent_mode').set_value('Direct response parameters').run()
        at.slider(key='ast').set_value(.1).run()
        assert not at.exception
        assert any('Using αst' in i.value for i in at.info)
        at.radio(key='spectrum_axis').set_value('Energy / eV').run()
        at.checkbox(key='compare').check().run()
        at.button(key='map').click().run()
        assert not at.exception
        assert len(at.session_state['hl_angle_map'][1])==361

    def test_random_initialization_persists_and_spacing_tracks_exchange(self):
        at=AppTest.from_file(str(APP),default_timeout=30).run()
        assert not at.exception
        original=[at.slider(key=f'angle_{i}').value for i in range(4)]
        assert all(0 < a < 90 for a in original)
        at.slider(key='angle_0').set_value(13.).run()
        assert [at.slider(key=f'angle_{i}').value for i in range(4)]==[13.,*original[1:]]
        at.slider(key='kmax').set_value(.8).run()
        assert not at.exception
        assert at.number_input(key='d_occ').value >= 1.65
        assert at.number_input(key='d_vir').value >= 1.65
        at.button(key='randomize_angles').click().run()
        assert not at.exception
        assert all(0 < at.slider(key=f'angle_{i}').value < 90 for i in range(4))

    def test_thermal_width_solvent_defaults_and_invalid_variance(self):
        at=AppTest.from_file(str(APP),default_timeout=30).run()
        assert not at.exception
        assert at.slider(key='n').value == 1.4
        assert at.slider(key='epsilon').value == 2.
        assert all(slider.key != 'fwhm' for slider in at.slider)
        before=next(c.value for c in at.caption if 'FWHM =' in c.value)
        at.slider(key='epsilon').set_value(12.).run()
        assert not at.exception
        assert next(c.value for c in at.caption if 'FWHM =' in c.value)!=before
        at.checkbox(key='compare').check().run()
        assert not at.exception
        assert any('Omitted comparison curves' in i.value for i in at.info)
        at.selectbox(key='molecule').select('E · negative solvatochromism').run()
        assert not at.exception
        assert any('nonpositive σ²' in w.value for w in at.warning)
