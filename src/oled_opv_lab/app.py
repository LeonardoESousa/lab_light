"""Two independent classroom activities. Run through oled-opv-lab."""
import secrets
import json
from dataclasses import asdict, replace
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from oled_opv_lab.models import OrbitalParameters, solve, composition, eligible, minimum_frontier_spacing
from oled_opv_lab.solvation import MOLECULES, HC, response, emission_energy, emission_sigma, sigma_to_fwhm, TEMPERATURE_K, gaussian_energy, wavelength_spectrum
from oled_opv_lab.colours import spectrum_to_color, vial_svg

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
    st.subheader('1 · Design a small singlet–triplet gap')
    st.write('Move the orbitals between donor and acceptor. Explore how localization and correlation change the lowest singlet and triplet states.')
    controls, plots = st.columns([1, 2.4], gap='large')
    with controls:
        st.markdown('**Orbital localization**')
        st.caption('Each angle describes where one orbital is localized: 0° = donor, 90° = acceptor, 45° = equal populations. It is not a torsion angle.')
        with st.expander('How do the localization angles work?'):
            st.latex(r'P_D=\cos^2\theta,\qquad P_A=\sin^2\theta')
            st.write('For example, 30° means 75% donor and 25% acceptor; 60° reverses these populations. The angle controls donor/acceptor amplitudes in the interaction model. An angle twice as large does not mean twice as much acceptor population. This descriptor does not specify the geometry or the orientation of an orbital in space.')
        # Initialize independently once; normal reruns must preserve the student's work.
        for i in range(4):
            st.session_state.setdefault(f'angle_{i}', float(secrets.randbelow(89)+1))
        if st.button('Randomize angles', key='randomize_angles'):
            for i in range(4):
                st.session_state[f'angle_{i}'] = float(secrets.randbelow(89)+1)
        angles = tuple(st.slider(f'θ({label}) / degrees', 0., 90., step=1., key=f'angle_{i}') for i,label in enumerate(LABELS))
        st.caption('0° = donor; 90° = acceptor. Angles describe localization, not a molecular torsion.')
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
        st.caption('Challenge: minimize |E(S1) − E(T1)| while BOTH lowest excited states retain ≥80% H→L character. Negative signed gaps indicate inversion.')
    energies = (-.2528, -.2528+d_occ, -.2528+d_occ+d_gap, -.2528+d_occ+d_gap+d_vir)
    p = OrbitalParameters(angles, energies, k, cross)
    r = solve(p)
    with plots:
        a,b,c = st.columns(3)
        a.metric('ΔEST = E(S1) − E(T1)', f"{1000*r['gap']:+.2f} meV")
        b.metric('H→L character · S1 / T1', f"{100*r['s1_hl']:.1f}% / {100*r['t1_hl']:.1f}%")
        c.metric('S2 − S1', f"{r['s2']-r['s1']:.4f} eV")
        valid = eligible(r)
        if r['t1'] <= 0:
            st.warning('The triplet lies below the lowest singlet: the assumed singlet-ground-state picture has broken down.')
        elif not valid:
            st.warning('Not eligible: both S1 and T1 must have at least 80% H→L character. Increase neighboring-orbital spacings or change localization to reduce state mixing.')
        else:
            st.success('Eligible attempt · '+('inverted singlet–triplet ordering' if r['gap'] < -1e-10 else 'normal or degenerate singlet–triplet ordering'))
        left,right = st.columns(2)
        donor = np.cos(np.radians(angles))**2
        fig = go.Figure()
        fig.add_bar(name='Donor', y=LABELS, x=donor, orientation='h', marker_color='#087e8b')
        fig.add_bar(name='Acceptor', y=LABELS, x=1-donor, orientation='h', marker_color='#e79947')
        fig.update_layout(barmode='stack')
        fig.update_xaxes(range=[0,1], tickformat='.0%')
        fig.update_yaxes(autorange='reversed')
        left.plotly_chart(style(fig,'Orbital localization','Population fraction',outside_legend=True), width='stretch')
        fig = go.Figure()
        for name, x, color in [('S1',0,'#087e8b'),('S2',1,'#41adb8'),('T1',2,'#a253ac')]:
            e = r[name.lower()]
            fig.add_scatter(x=[x-.3,x+.3],y=[e,e], mode='lines', line=dict(color=color,width=5), name=f'{name}: {e:.4f} eV')
        fig.update_xaxes(tickvals=[0,1,2],ticktext=['S1','S2','T1'])
        levels = [r['s1'], r['s2'], r['t1']]
        padding = max(.01, .12 * (max(levels)-min(levels)))
        fig.update_yaxes(range=[min(levels)-padding, max(levels)+padding])
        right.plotly_chart(style(fig,'CISD excited-state levels',y='Energy above S0 / eV',outside_legend=True), width='stretch')
        with st.expander('What are these states made of?'):
            for state in ['S1','T1','S2']:
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
               'T1 H→L (%)':100*r['t1_hl'],'Eligible':valid}
        st.session_state.setdefault('hl_attempts',[]).append(row)
    attempts = st.session_state.get('hl_attempts',[])
    if attempts:
        df = pd.DataFrame(attempts)
        eligible_rows = df[df['Eligible']]
        if len(eligible_rows):
            best = eligible_rows.loc[eligible_rows['ΔEST / meV'].abs().idxmin()]
            st.info(f"Your closest eligible gap this session: {best['ΔEST / meV']:+.2f} meV")
        st.dataframe(df, hide_index=True, width='stretch')
        st.download_button('Download attempts (CSV)',df.to_csv(index=False).encode('utf-8'), 'orbital_attempts.csv','text/csv')
        if st.button('Clear attempts'):
            st.session_state['hl_attempts'] = []
            st.rerun()
    with st.expander('Map the design space'):
        st.write('Scan H−1 and L+1 angles with the other parameters fixed. The map is calculated only when requested.')
        if st.button('Calculate angle map', key='map'):
            with st.spinner('Diagonalizing singlet and triplet Hamiltonians…'):
                grid = np.linspace(0,90,19)
                rows=[]
                for b in grid:
                    for a in grid:
                        q = replace(p,angles=(float(a),angles[1],angles[2],float(b)))
                        rr=solve(q)
                        rows.append([a,b,1000*rr['gap'],100*rr['s1_hl'],100*rr['t1_hl'],eligible(rr)])
                st.session_state['hl_angle_map']=(p,rows)
        saved = st.session_state.get('hl_angle_map')
        if saved:
            old, rows=saved
            if old != p:
                st.info('Parameters changed. Recalculate to update the map.')
            else:
                frame=pd.DataFrame(rows,columns=['θ(H−1)','θ(L+1)','ΔEST / meV','S1 H→L (%)','T1 H→L (%)','Eligible'])
                z=frame['ΔEST / meV'].to_numpy().reshape(19,19)
                mask=~frame['Eligible'].to_numpy(dtype=bool)
                z=np.where(mask.reshape(19,19),np.nan,z)
                fig=go.Figure(go.Heatmap(x=np.linspace(0,90,19),y=np.linspace(0,90,19),z=z,colorscale='RdBu_r',zmid=0,colorbar_title='meV',hoverongaps=False))
                st.plotly_chart(style(fig,'Signed singlet–triplet gap','θ(H−1) / degrees','θ(L+1) / degrees',450),width='stretch')
                st.caption('Blank cells fail the current eligibility criteria.')
                st.download_button('Download map (CSV)',frame.to_csv(index=False).encode(),'angle_map.csv','text/csv')
    st.download_button('Download current model settings',json.dumps(asdict(p),indent=2),'orbital_settings.json','application/json')
    with st.expander('Model and assumptions'):
        st.write('The H−1→L and H→L+1 configurations compete with H→L when neighboring orbitals are too close. Their bare singlet ordering depends on neighboring-orbital spacings versus exchange differences (up to 2Kmax), not on a universal lower HOMO–LUMO gap. The spacing floor is a conservative bare-ordering safeguard. Correlation can still mix configurations: eligibility therefore requires ≥80% H→L weight in BOTH actual lowest roots, without relabeling higher roots as S1/T1.')
        st.latex(r'u_p=(\cos\theta_p,\sin\theta_p),\quad P_D=\cos^2\theta_p,\quad P_A=\sin^2\theta_p')
        st.latex(r'g_{pqrs}=K_{\max}\sum_{X,Y\in\{D,A\}}u_{pX}u_{qX}M_{XY}u_{rY}u_{sY},\quad M=\begin{pmatrix}1&\lambda\\\lambda&1\end{pmatrix}')
        st.write('This retains the earlier effective model: canonical orbital energies, exchange-only configuration blocks, localization-dependent couplings, and zero reference–single coupling. Direct Coulomb differences and orbital relaxation are omitted. The two-component amplitudes are localization descriptors, not four orthonormal orbitals in a two-dimensional basis.')
        st.write('Full diagonalization within CISD: 15 singlet CSFs (reference + 4 singles + 10 doubles) and 11 triplet CSFs (4 singles + 7 doubles). No triples or quadruples; this is not full CI. All excitation energies use the correlated lowest singlet S0 as zero. This qualitative model does not predict actual molecular energies or emission brightness.')


def solvation_activity():
    st.subheader('2 · Tune emission with the solvent')
    st.write('Keep the molecule fixed and change the solvent response. Can you rank the molecules by their sensitivity before comparing the spectra?')
    left,right=st.columns([1,2.4],gap='large')
    with left:
        selected=st.selectbox('Test molecule',[m.name for m in MOLECULES]+['Custom molecule'],key='molecule')
        if selected=='Custom molecule':
            ev=st.number_input('Vacuum emission energy / eV',1.,5.,2.9,.05,key='ev')
            chi=st.slider('χ / eV',-1.5,1.5,.65,.01,key='chi')
        else:
            mol=next(m for m in MOLECULES if m.name==selected)
            ev,chi=mol.e_vac,mol.chi
            st.caption(f'E_vac = {ev:.2f} eV · χ = {chi:+.2f} eV')
        st.caption('These are fictional teaching molecules, not experimental fits.')
        mode=st.radio('Solvent controls',['Dielectric constant and refractive index','Direct response parameters'],key='solvent_mode')
        if mode=='Dielectric constant and refractive index':
            n=st.slider('Refractive index n',1.,1.8,1.4,.01,key='n')
            # Fixed bounds avoid changing widget identities when n changes.
            epsilon_input=st.slider('Static dielectric constant εs',1.,20.,2.,.1,key='epsilon')
            epsilon=max(epsilon_input,n*n)
            if epsilon_input<n*n:
                st.info(f'Using εs = n² = {n*n:.3f}; the static response must be at least the optical response.')
            ast,aopt=response(epsilon),response(n*n)
        else:
            aopt=st.slider('Optical response αopt',0.,.95,float(response(1.4**2)),.005,key='aopt')
            ast_input=st.slider('Static response αst',0.,.95,float(response(2.)),.005,key='ast')
            ast=max(ast_input,aopt)
            if ast_input<aopt:
                st.info('Using αst = αopt to keep the static response at least as large as the optical response.')
        st.caption('Temperature fixed at 300 K; the model calculates the linewidth.')
        st.latex(r'\sigma=\sqrt{(k_BT)^2+2\chi(\alpha_{\rm st}-\alpha_{\rm opt})k_BT}')
        axis=st.radio('Spectrum axis',['Wavelength / nm','Energy / eV'],key='spectrum_axis')
        overlay=st.checkbox('Compare vacuum spectrum',True,key='vacuum')
        compare=st.checkbox('Compare all test molecules',False,key='compare')
    try:
        peak=emission_energy(ev,chi,ast,aopt)
        sigma=emission_sigma(chi,ast,aopt)
    except ValueError as error:
        with right:
            st.warning(str(error))
        return
    width=sigma_to_fwhm(sigma)
    vacuum_width=sigma_to_fwhm(emission_sigma(chi,0.,0.))
    # Full range for colour calculation prevents normalization of a tiny UV/IR tail.
    wavelengths=np.linspace(200,2500,4601)
    intensity=wavelength_spectrum(wavelengths,peak,width)
    rgb=spectrum_to_color(wavelengths,intensity)
    with right:
        a,b,c=st.columns(3)
        a.metric('Emission centre',f'{peak:.3f} eV')
        b.metric('Energy-centre wavelength',f'{HC/peak:.0f} nm')
        c.metric('Solvent shift',f'{1000*(peak-ev):+.0f} meV')
        chart,vial=st.columns([3,1])
        is_nm=axis=='Wavelength / nm'
        x=np.linspace(300,1000,1401) if is_nm else np.linspace(.5,5.,1401)
        def density(p, fwhm=width):
            return wavelength_spectrum(x,p,fwhm) if is_nm else gaussian_energy(x,p,fwhm)
        fig=go.Figure()
        color=f'rgb{rgb}'
        fig.add_scatter(x=x,y=density(peak),name='Selected molecule · solvent',line=dict(color=color,width=4),fill='tozeroy')
        if overlay:
            fig.add_scatter(x=x,y=density(ev,vacuum_width),name='Selected molecule · vacuum',line=dict(color='#8492a6',dash='dash',width=2))
        skipped=[]
        if compare:
            for m in MOLECULES:
                if m.name!=selected:
                    try:
                        m_peak=emission_energy(m.e_vac,m.chi,ast,aopt)
                        m_width=sigma_to_fwhm(emission_sigma(m.chi,ast,aopt))
                    except ValueError:
                        skipped.append(m.name)
                        continue
                    fig.add_scatter(x=x,y=density(m_peak,m_width),name=m.name,line=dict(width=2))
        chart.plotly_chart(style(fig,'Emission spectrum',axis,'Spectral density / '+('nm⁻¹' if is_nm else 'eV⁻¹'),520 if compare else 450,outside_legend=True),width='stretch')
        vial.image(vial_svg(rgb),width='stretch')
        vial.caption('Colour uses the entire emission spectrum, including its width. Brightness is normalized.')
        if skipped:
            st.info('Omitted comparison curves with invalid model energy or variance: '+', '.join(skipped))
        st.caption(f'300 K · σ = {sigma:.4f} eV · FWHM = {width:.4f} eV')
        st.caption('At fixed centre energy, broader emission generally gives a less saturated colour and may also change its hue. The vial is an approximate screen rendering, not a brightness or colour-purity measurement.')
        st.caption(f'αst = {ast:.4f} · αopt = {aopt:.4f} · 2αst − αopt = {2*ast-aopt:.4f}')
        st.caption('The Gaussian is defined per eV. Wavelength spectra include the Jacobian |dE/dλ|, so their maxima need not occur exactly at hc/Ecentre.')
        if not .38e3<HC/peak<.78e3:
            st.info('The emission centre is outside the visible range; the vial colour represents the visible part only.')
        data=pd.DataFrame({'Wavelength / nm':wavelengths,'Solvent density / nm^-1':intensity,
                           'Vacuum density / nm^-1':wavelength_spectrum(wavelengths,ev,vacuum_width),
                           'Solvent sigma / eV':sigma, 'Solvent FWHM / eV':width,
                           'Temperature / K':TEMPERATURE_K, 'alpha_st':ast, 'alpha_opt':aopt})
        st.download_button('Download spectra (CSV)',data.to_csv(index=False).encode(),'solvatochromism_spectra.csv','text/csv')
    with st.expander('Solvent response and model'):
        st.latex(r'E_{\rm sol}=E_{\rm vac}-\chi(2\alpha_{\rm st}-\alpha_{\rm opt})')
        st.latex(r'\alpha_{\rm st}=\frac{\varepsilon_s-1}{\varepsilon_s+1},\qquad\alpha_{\rm opt}=\frac{n^2-1}{n^2+1}')
        st.write('The response convention is defined here explicitly because α has several conventions in the literature. χ is an effective sensitivity in eV; it absorbs cavity and molecular factors. Positive χ produces a red shift as static polarity increases at fixed optical response; negative χ produces a blue shift. At 300 K, σ²=(kBT)²+2χ(αst−αopt)kBT, with kB in eV/K. For positive χ, greater static–optical response difference broadens the Gaussian. χ=0 or αst=αopt leaves σ=kBT≈0.02585 eV. This is a teaching model connecting greater solvent sensitivity (often associated with CT character) to broader emission; CT character alone does not guarantee a broad spectrum. Integrated energy-domain intensity stays constant; specific interactions, aggregation and changes in quantum yield are omitted. Negative χ may yield an invalid variance; those combinations are reported rather than silently modified.')
        st.write('The vial adapts NEMOview’s CIE 1931 / D65 / sRGB pipeline with corrected normalization and proportional geometry. It is an illustrative screen colour; D65 weighting is retained for continuity with that code, rather than a calibrated self-luminous emission prediction.')
    st.markdown('**Try it:** Find two molecule–solvent combinations with similar emission energies. Can they still have different solvent sensitivities? Why does changing refractive index at fixed εs shift the spectrum in the opposite direction to increasing εs?')


def main():
    st.set_page_config(page_title='Molecular Light Lab',page_icon='🧪',layout='wide')
    st.title('Molecular Light Lab')
    st.caption('OLEDs & OPVs · Explore molecular properties, predict an outcome, then test it.')
    first,second=st.tabs(['1 · Four-orbital CISD','2 · Solvatochromism'])
    with first:
        orbital_activity()
    with second:
        solvation_activity()
    st.caption('Teaching models · version 0.1.0 · Saved attempts persist only during your current browser session. Download them before leaving.')


if __name__=='__main__':
    main()
