"""Two independent classroom activities. Run through oled-opv-lab."""
import secrets
import json
from dataclasses import asdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from oled_opv_lab.models import OrbitalParameters, solve, composition, minimum_frontier_spacing
from oled_opv_lab.solvation import localization_properties, HC, response, emission_energy, emission_peak, emission_sigma, sigma_to_fwhm, TEMPERATURE_K, gaussian_energy, wavelength_spectrum
from oled_opv_lab.colours import spectrum_to_color, scale_emission_color, vial_svg
from oled_opv_lab.visuals import orbital_density_figure
from oled_opv_lab.opv_ui import opv_activity

LABELS = ['H−1', 'H', 'L', 'L+1']


def style(fig, title, x='', y='', height=350, outside_legend=False):
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, height=height,
                      font=dict(family='Arial, sans-serif', size=16),
                      margin=dict(l=35,r=30,t=55,b=40),
                      legend=dict(x=.02,y=.98), template='plotly_white')
    if outside_legend:
        fig.update_layout(legend=dict(orientation='h', x=0, xanchor='left',
                                      y=-.28, yanchor='top'), margin=dict(b=120))
    return fig


def orbital_activity():
    st.subheader('1 · Singlet-triplet gaps')
    st.write('Move the orbitals between donor and acceptor. Explore how localization and correlation change the lowest singlet and triplet states.')
    controls, plots = st.columns([1, 2.4], gap='large')
    with controls:
        st.markdown('**Orbital localization**')
        st.markdown('**θ = 0° means localization in D (donor).**')
        st.markdown('**θ = 90° means localization in A (acceptor).**')
        st.caption('At intermediate angles the orbital is distributed between D and A; θ = 45° gives equal populations.')
        with st.expander('Localization populations'):
            st.latex(r'P_D=\cos^2\theta,\qquad P_A=\sin^2\theta')
            st.write('At θ = 30°, the orbital has 75% donor and 25% acceptor population. The angles describe localization, not a molecular torsion or orbital orientation.')
        # Initialize independently once; normal reruns must preserve the student's work.
        for i in range(4):
            st.session_state.setdefault(f'angle_{i}', float(secrets.randbelow(89)+1))
        if st.button('Randomize angles', key='randomize_angles'):
            for i in range(4):
                st.session_state[f'angle_{i}'] = float(secrets.randbelow(89)+1)
        angles = tuple(st.slider(f'θ({label}) / degrees', 0., 90., step=1., key=f'angle_{i}') for i,label in enumerate(LABELS))
        with st.expander('Orbital energies and interaction'):
            k = st.slider('Kmax / eV', 0., 1., .1, .01, key='kmax')
            cross = st.slider('Cross-fragment interaction λ', 0., 1., 0., .01, key='cross')
            spacing_floor = minimum_frontier_spacing(k)
            for key, default in [('d_occ', .5), ('d_vir', .5759)]:
                st.session_state[key] = max(float(st.session_state.get(key, default)), spacing_floor)
            d_occ = st.number_input('ε(H) − ε(H−1) / eV', min_value=spacing_floor, max_value=5., step=.01, format='%.4f', key='d_occ')
            d_gap = st.slider('ε(L) − ε(H) / eV', .1, 10., 3.026, .001, key='d_gap')
            d_vir = st.number_input('ε(L+1) − ε(L) / eV', min_value=spacing_floor, max_value=5., step=.01, format='%.4f', key='d_vir')
            st.caption(f'Neighboring-orbital spacings ≥ 2Kmax + 0.05 eV = {spacing_floor:.3f} eV. This protects bare H→L ordering; CISD character is checked below.')
        st.caption('Explore the signed S1–T1 gap. The H→L percentages show whether the two lowest states still describe the intended transition.')
    energies = (-.2528, -.2528+d_occ, -.2528+d_occ+d_gap, -.2528+d_occ+d_gap+d_vir)
    p = OrbitalParameters(angles, energies, k, cross)
    r = solve(p)
    with plots:
        a,b = st.columns(2)
        a.metric('ΔEST = E(S1) − E(T1)', f"{1000*r['gap']:+.2f} meV")
        b.metric('H→L character · S1 / T1', f"{100*r['s1_hl']:.1f}% / {100*r['t1_hl']:.1f}%")
        left,right = st.columns(2)
        fig=orbital_density_figure(angles)
        left.plotly_chart(style(fig,'Orbital localization',height=410),width='stretch')
        left.caption('Schematic density inside fixed D/A halves. L+1 is highest; H−1 is lowest. Lobe areas follow donor/acceptor populations; row spacing is schematic.')
        fig = go.Figure()
        for name, x, color in [('S1',0,'#087e8b'),('T1',1,'#a253ac')]:
            e = r[name.lower()]
            fig.add_scatter(x=[x-.3,x+.3],y=[e,e], mode='lines', line=dict(color=color,width=5), name=f'{name}: {e:.4f} eV')
        fig.update_xaxes(tickvals=[0,1],ticktext=['S1','T1'])
        levels = [r['s1'], r['t1']]
        padding = max(.01, .12 * (max(levels)-min(levels)))
        fig.update_yaxes(range=[min(levels)-padding, max(levels)+padding])
        right.plotly_chart(style(fig,'CISD excited-state levels',y='Energy above S0 / eV',outside_legend=True), width='stretch')
        with st.expander('What are these states made of?'):
            for state in ['S1','T1']:
                df = pd.DataFrame(composition(r,state)).groupby(['Configuration','Rank'],as_index=False).sum().sort_values('Weight (%)',ascending=False)
                st.markdown(f'**{state}**')
                st.dataframe(df, hide_index=True, width='stretch')
        st.caption(f"S0 reference weight: {100*r['s0_reference']:.1f}%. State labels follow energy order; their molecular character can change at crossings.")
    if st.button('Save this attempt', key='save_orbital'):
        row = {**{f'θ({l}) / deg':a for l,a in zip(LABELS,angles)},
               'Kmax / eV':k,'λ':cross,'occupied spacing / eV':d_occ,
               'H–L spacing / eV':d_gap,'virtual spacing / eV':d_vir,
               'S1 / eV':r['s1'],'T1 / eV':r['t1'],'ΔEST / meV':1000*r['gap'],
               'S1 singles (%)':100*r['s1_single'],'S1 H→L (%)':100*r['s1_hl'],
               'T1 H→L (%)':100*r['t1_hl']}
        st.session_state.setdefault('st_attempts',[]).append(row)
    attempts = st.session_state.get('st_attempts',[])
    if attempts:
        df = pd.DataFrame(attempts)
        st.dataframe(df, hide_index=True, width='stretch')
        st.download_button('Download attempts (CSV)',df.to_csv(index=False).encode('utf-8'), 'orbital_attempts.csv','text/csv')
        if st.button('Clear attempts'):
            st.session_state['st_attempts'] = []
            st.rerun()
    st.download_button('Download current model settings',json.dumps(asdict(p),indent=2),'orbital_settings.json','application/json')
    with st.expander('Model and assumptions'):
        st.write('The H−1→L and H→L+1 configurations compete with H→L when neighboring orbitals are too close. Their bare singlet ordering depends on neighboring-orbital spacings versus exchange differences (up to 2Kmax), not on a universal lower HOMO–LUMO gap. The spacing floor is a conservative bare-ordering safeguard. Correlation can still mix configurations. The displayed H→L percentages describe the actual lowest roots; higher roots are not relabeled as S1/T1.')
        st.latex(r'u_p=(\cos\theta_p,\sin\theta_p),\quad P_D=\cos^2\theta_p,\quad P_A=\sin^2\theta_p')
        st.latex(r'g_{pqrs}=K_{\max}\sum_{X,Y\in\{D,A\}}u_{pX}u_{qX}M_{XY}u_{rY}u_{sY},\quad M=\begin{pmatrix}1&\lambda\\\lambda&1\end{pmatrix}')
        st.write('This retains the earlier effective model: canonical orbital energies, exchange-only configuration blocks, localization-dependent couplings, and zero reference–single coupling. Direct Coulomb differences and orbital relaxation are omitted. The two-component amplitudes are localization descriptors, not four orthonormal orbitals in a two-dimensional basis.')
        st.write('Full diagonalization within CISD: 15 singlet CSFs (reference + 4 singles + 10 doubles) and 11 triplet CSFs (4 singles + 7 doubles). No triples or quadruples; this is not full CI. All excitation energies use the correlated lowest singlet S0 as zero. This qualitative model does not predict actual molecular energies or emission brightness.')


def solvation_activity():
    st.subheader('2 · Solvatochromism')
    st.write('Move the HOMO and LUMO between D and A, then change the solvent. Compare the emission energy, width and intensity.')
    left,right=st.columns([1,2.4],gap='large')
    with left:
        st.markdown('**HOMO and LUMO localization**')
        st.caption('θ = 0°: localized in D (donor). θ = 90°: localized in A (acceptor). θ = 45°: equal D/A populations.')
        theta_h=st.slider('θ(HOMO) / degrees',0.,90.,20.,1.,key='solv_theta_h')
        theta_l=st.slider('θ(LUMO) / degrees',0.,90.,65.,1.,key='solv_theta_l')
        props=localization_properties(theta_h,theta_l)
        chi,strength=props['chi'],props['strength']
        st.caption(f"Localization overlap proxy = {props['overlap']:.3f} · relative emission strength = {strength:.3f}")
        ev=st.number_input('Vacuum emission energy / eV',1.,5.,2.9,.05,key='ev')
        st.markdown('**Solvent controls**')
        n=st.slider('Refractive index n',1.,1.8,1.4,.01,key='n')
        epsilon_input=st.slider('Static dielectric constant εs',1.,20.,2.,.1,key='epsilon')
        epsilon=max(epsilon_input,n*n)
        if epsilon_input<n*n:
            st.info(f'Using εs = n² = {n*n:.3f}; the static response must be at least the optical response.')
        ast,aopt=response(epsilon),response(n*n)
        st.caption('Temperature fixed at 300 K; the model calculates the linewidth.')
        axis=st.radio('Spectrum axis',['Wavelength / nm','Energy / eV'],key='spectrum_axis')
        normalize=st.checkbox('Normalize each spectrum (compare shapes)',True,key='normalize_spectra')
        overlay=st.checkbox('Compare vacuum spectrum',True,key='vacuum')
    try:
        peak=emission_energy(ev,chi,ast,aopt)
        sigma=emission_sigma(chi,ast,aopt)
    except ValueError as error:
        with right:
            st.warning(str(error))
        return
    width=sigma_to_fwhm(sigma)
    vacuum_width=sigma_to_fwhm(emission_sigma(chi,0.,0.))
    wavelengths=np.linspace(200,2500,4601)
    shape=wavelength_spectrum(wavelengths,peak,width)
    intensity=strength*shape
    vacuum_intensity=strength*wavelength_spectrum(wavelengths,ev,vacuum_width)
    # Integrate the whole shape for hue, then retain the intensity scaling for brightness.
    rgb=spectrum_to_color(wavelengths,shape)
    vial_rgb=scale_emission_color(rgb,strength)
    with right:
        a,b,c=st.columns(3)
        peak_ev,peak_nm=emission_peak(peak,sigma,wavelength_axis=axis=='Wavelength / nm')
        a.metric('Emission peak / eV',f'{peak_ev:.3f}')
        b.metric('Emission peak / nm',f'{peak_nm:.0f}')
        c.metric('Relative emission strength',f'{strength:.3f}')
        populations=orbital_density_figure((theta_h,theta_l),labels=('HOMO','LUMO'))
        st.plotly_chart(style(populations,'HOMO / LUMO localization',height=300),width='stretch')
        chart,vial=st.columns([3,1])
        is_nm=axis=='Wavelength / nm'
        x=np.linspace(300,1000,1401) if is_nm else np.linspace(.5,5.,1401)
        def density(p, fwhm):
            y=wavelength_spectrum(x,p,fwhm) if is_nm else gaussian_energy(x,p,fwhm)
            y=strength*y
            if normalize and y.max()>0:
                y=y/y.max()
            return y
        solvent_curve=density(peak,width)
        vacuum_curve=density(ev,vacuum_width)
        fig=go.Figure()
        fig.add_scatter(x=x,y=solvent_curve,name='Solvent',line=dict(color=f'rgb{rgb}',width=4),fill='tozeroy')
        if overlay:
            fig.add_scatter(x=x,y=vacuum_curve,name='Vacuum',line=dict(color='#8492a6',dash='dash',width=2))
        if normalize:
            fig.update_yaxes(range=[0,1.08])
        else:
            # Relative intensities are preserved; fit the displayed curves instead
            # of reserving space for a hypothetical unit-strength narrow peak.
            maximum=max(float(solvent_curve.max()),float(vacuum_curve.max()) if overlay else 0.)
            fig.update_yaxes(range=[0,1.08*maximum if maximum>0 else 1.])
        chart.plotly_chart(style(fig,'Emission spectrum',axis,'Intensity (a.u.)',450,outside_legend=True),width='stretch')
        vial.image(vial_svg(vial_rgb),width='stretch')
        vial.caption('Colour uses the full spectrum; brightness follows relative emission strength.')
        if strength==0:
            st.caption('No emission in this idealized zero-overlap limit. The calculated transition energy describes a dark state.')
        st.caption(f'300 K · σ = {sigma:.4f} eV · FWHM = {width:.4f} eV')
        st.caption(f'αst = {ast:.4f} · αopt = {aopt:.4f} · solvent shift = {1000*(peak-ev):+.0f} meV')
        if normalize:
            st.caption('Shape comparison: each nonzero spectrum is normalized to its own visible-window maximum of 1. Peak heights do not compare brightness. Relative emission strength and vial brightness retain the overlap dependence; downloaded spectra retain their physical scaling.')
        else:
            st.caption('Relative-intensity view: spectral area scales with the overlap proxy, and broadening reduces peak height. The y axis fits the displayed curves. Wavelength curves include |dE/dλ|.')
        st.caption('Peak values refer to the maximum of the spectrum on the selected axis; the eV and nm values describe the same photon energy.')
        if not 380<peak_nm<780:
            st.info('The emission peak is outside the visible range; the vial represents the visible emission only.')
        data=pd.DataFrame({'Wavelength / nm':wavelengths,'Solvent intensity / a.u.':intensity,
                           'Vacuum intensity / a.u.':vacuum_intensity,
                           'HOMO angle / deg':theta_h,'LUMO angle / deg':theta_l,
                           'Localization overlap proxy':props['overlap'],'chi / eV':chi,
                           'Relative emission strength':strength,'Solvent sigma / eV':sigma,
                           'Solvent FWHM / eV':width,'Temperature / K':TEMPERATURE_K,
                           'alpha_st':ast,'alpha_opt':aopt})
        st.download_button('Download spectra (CSV)',data.to_csv(index=False).encode(),'solvatochromism_spectra.csv','text/csv')
    with st.expander('Localization, solvent response and model'):
        st.latex(r'O=\cos^2(\theta_H-\theta_L),\qquad \chi=\chi_{\max}(1-O),\qquad I_{\rm rel}=O')
        st.write(f'The localization overlap proxy compares the D/A descriptors of the HOMO and LUMO. Equal angles give O=1: small solvent sensitivity and strong emission. Opposite localization in D and A gives O=0: maximum solvent sensitivity and zero emission strength in this idealized model..')
        st.write('These mappings are teaching assumptions, not predictions of transition dipoles or measured quantum yields. O is not the overlap integral of the actual molecular orbitals, which remain orthogonal. Equal brightness excitation conditions are assumed; nonradiative competition and selection rules are not modeled.')
        st.latex(r'E_{\rm sol}=E_{\rm vac}-\chi(2\alpha_{\rm st}-\alpha_{\rm opt})')
        st.latex(r'\alpha_{\rm st}=\frac{\varepsilon_s-1}{\varepsilon_s+1},\qquad\alpha_{\rm opt}=\frac{n^2-1}{n^2+1}')
        st.latex(r'\sigma=\sqrt{(k_BT)^2+2\chi(\alpha_{\rm st}-\alpha_{\rm opt})k_BT}')
        st.write('The Gaussian is defined per eV; FWHM=2√(2ln2)σ. The vacuum has σ=kBT. Larger χ increases broadening when αst>αopt. The vial uses the full spectrum for its approximate CIE/D65/sRGB colour and scales linear-light brightness with Irel. Screen gamut and clipping can hide some colour changes; this is not a calibrated colour-purity measurement.')
    st.markdown('**Try it:** Separate the HOMO and LUMO. Does the emission shift more strongly with solvent polarity? What happens to its width, integrated intensity and colour?')

def main():
    st.set_page_config(page_title='Molecular Light Lab',page_icon='🧪',layout='wide')
    st.title('Molecular Light Lab')
    st.caption('OLEDs & OPVs · Explore molecular properties, predict an outcome, then test it.')
    first,second,third=st.tabs(['Singlet-triplet gaps','Solvatochromism','OPV · Harvest sunlight'])
    with first:
        orbital_activity()
    with second:
        solvation_activity()
    with third:
        opv_activity(style)
    st.caption('Teaching models · version 0.1.0 · Saved attempts persist only during your current browser session. Download them before leaving.')


if __name__=='__main__':
    main()
