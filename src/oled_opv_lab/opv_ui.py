"""Student-facing OPV spectral matching activity."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from .opv import (OPVParameters,calculate_opv,HC_EV_NM,FWHM_EV,
                  ABSORPTANCE_MAX,SEPARATION_SCALE_EV,LOSS_EV,FILL_FACTOR)


def opv_activity(style):
    st.subheader('3 · Harvest sunlight')
    st.write('Tune donor absorption and its alignment with an acceptor. First maximize absorbed photons; then try to maximize the electrical power score.')
    controls,display=st.columns([1,2.4],gap='large')
    with controls:
        gap=st.slider('Donor absorption peak / eV',1.,3.5,1.9,.01,key='opv_gap')
        st.caption(f'Absorption maximum: {HC_EV_NM/gap:.0f} nm. The peak energy is treated as an effective donor gap.')
        offset=st.slider('Donor–acceptor LUMO offset Δ / eV',0.,1.,.3,.01,key='opv_offset')
        st.caption('Δ = LUMO(D) − LUMO(A). The donor HOMO stays fixed; the acceptor LUMO follows the selected offset.')
        st.caption(f'Fixed Gaussian FWHM: {FWHM_EV:.2f} eV · peak absorptance: {100*ABSORPTANCE_MAX:.0f}% · fill factor: {FILL_FACTOR:.2f}.')
        st.markdown('**Try two objectives**')
        st.write('1. Find an absorption peak that captures many photons.\n2. Adjust the energy offset and absorption peak to improve power. Is the best photon absorber also the best power producer?')
    result=calculate_opv(OPVParameters(gap_ev=gap,offset_ev=offset))
    r=result
    with display:
        a,b,c,d=st.columns(4)
        a.metric('Absorbed photons / m²/s',f"{r['absorbed_photons_m2_s']:.2e}")
        b.metric('Charge-generation yield',f"{100*r['separation_yield']:.1f}%")
        c.metric('Jsc estimate / mA/cm²',f"{r['jsc_ma_cm2']:.2f}")
        d.metric('Voc estimate / V',f"{r['voc_v']:.2f}")
        st.metric('Power score / W/m²',f"{r['power_w_m2']:.2f}")
        st.caption('Power score = Jsc × Voc × fixed fill factor. These are simplified model estimates, not a calibrated device efficiency.')
        w=r['wavelength_nm']
        fig=go.Figure()
        fig.add_scatter(x=w,y=r['photon_flux_nm']/1e18,name='AM1.5G incident photons',line=dict(color='#8b98a9',width=1.5))
        fig.add_scatter(x=w,y=r['absorbed_flux_nm']/1e18,name='Absorbed photons',line=dict(color='#087e8b',width=3),fill='tozeroy')
        fig.add_scatter(x=w,y=r['absorptance'],name='Donor absorptance',yaxis='y2',line=dict(color='#d88932',width=2.5,dash='dash'))
        fig=style(fig,'Solar photons and donor absorption','Wavelength / nm','Photon flux / 10¹⁸ m⁻² s⁻¹ nm⁻¹',470,outside_legend=True)
        fig.update_layout(yaxis2=dict(title='Absorptance',overlaying='y',side='right',range=[0,1],tickformat='.0%',showgrid=False),margin=dict(r=85))
        fig.update_xaxes(range=[300,1400])
        fig.update_yaxes(rangemode='tozero',selector=dict(anchor='x'))
        st.plotly_chart(fig,width='stretch')
        st.caption(f"Integration uses the entire bundled 280–4000 nm spectrum, not just the displayed window. Incident power: {r['incident_power_w_m2']:.1f} W/m². Absorbed photon fraction: {100*r['absorbed_fraction']:.1f}%.")
        levels=go.Figure()
        for name,x,energy,color in [('HOMO(D)',0,r['donor_homo_ev'],'#087e8b'),
                                     ('LUMO(D)',0,r['donor_lumo_ev'],'#087e8b'),
                                     ('HOMO(A)',1,r['acceptor_homo_ev'],'#d88932'),
                                     ('LUMO(A)',1,r['acceptor_lumo_ev'],'#d88932')]:
            levels.add_scatter(x=[x-.23,x+.23],y=[energy,energy],mode='lines',
                               line=dict(width=5,color=color),name=f'{name}: {energy:.2f} eV')
        levels.add_annotation(x=.77,y=r['acceptor_lumo_ev'],ax=.23,ay=r['donor_lumo_ev'],
                              xref='x',yref='y',axref='x',ayref='y',showarrow=True,
                              arrowhead=2,arrowwidth=2,text='Electron transfer',font=dict(size=13))
        levels.update_xaxes(tickvals=[0,1],ticktext=['Donor','Acceptor'],range=[-.4,1.4])
        st.plotly_chart(style(levels,'Donor–acceptor energy alignment',y='Energy relative to vacuum / eV',height=430,outside_legend=True),width='stretch')
        st.caption(f"Effective donor gap: {gap:.2f} eV · acceptor gap: {r['acceptor_lumo_ev']-r['acceptor_homo_ev']:.2f} eV · CT-state energy: {r['ct_energy_ev']:.2f} eV. Only donor absorption is modeled.")
    if st.button('Save OPV attempt',key='opv_save'):
        row={'Donor peak / eV':gap,'Donor peak / nm':HC_EV_NM/gap,'Offset / eV':offset,
             'Absorbed photons / m²/s':r['absorbed_photons_m2_s'],
             'Charge yield (%)':100*r['separation_yield'],'Jsc / mA/cm²':r['jsc_ma_cm2'],
             'Voc / V':r['voc_v'],'Power score / W/m²':r['power_w_m2']}
        st.session_state.setdefault('opv_attempts',[]).append(row)
    attempts=st.session_state.get('opv_attempts',[])
    if attempts:
        table=pd.DataFrame(attempts)
        st.caption(f"Best saved power score this session: {table['Power score / W/m²'].max():.2f} W/m²")
        st.dataframe(table,hide_index=True,width='stretch')
        st.download_button('Download OPV attempts (CSV)',table.to_csv(index=False).encode(),
                           'opv_attempts.csv','text/csv',key='opv_attempt_download')
        if st.button('Clear OPV attempts',key='opv_clear'):
            st.session_state['opv_attempts']=[]
            st.rerun()
    spectra=pd.DataFrame({'Wavelength / nm':r['wavelength_nm'],'AM1.5G / W m^-2 nm^-1':r['irradiance_w_m2_nm'],
                          'Incident photons / m^-2 s^-1 nm^-1':r['photon_flux_nm'],'Absorptance':r['absorptance'],
                          'Absorbed photons / m^-2 s^-1 nm^-1':r['absorbed_flux_nm'],
                          'Collected charges / m^-2 s^-1 nm^-1':r['collected_flux_nm']})
    st.download_button('Download OPV spectra (CSV)',spectra.to_csv(index=False).encode(),'opv_spectra.csv','text/csv',key='opv_spectrum_download')
    with st.expander('Model, assumptions and spectrum source'):
        st.latex(r'A(E)=0.85\exp\left[-\frac{(E-E_g)^2}{2\sigma^2}\right],\qquad\sigma=\frac{0.45\ {\rm eV}}{2\sqrt{2\ln2}}')
        st.latex(r'\Phi_\lambda=\frac{I_\lambda}{hc/\lambda},\qquad N_{\rm abs}=\int A(hc/\lambda)\Phi_\lambda\,d\lambda')
        st.write('The solar curve is photon flux, not irradiance: each absorbed photon counts once. Absorptance is dimensionless, so it is evaluated at E=hc/λ without a spectral-density Jacobian. There is no independent normalization of solar or absorbed photon curves.')
        st.latex(r'\eta_{\rm sep}=1-e^{-\Delta/\Delta_0},\qquad J_{\rm SC}=qN_{\rm abs}\eta_{\rm sep}')
        st.latex(r'E_{\rm CT}=E_g-\Delta,\qquad V_{\rm OC}\approx\max(E_{\rm CT}-E_{\rm loss},0)/q')
        st.write(f'Δ0={SEPARATION_SCALE_EV:.2f} eV, voltage-loss energy={LOSS_EV:.2f} eV, and FF={FILL_FACTOR:.2f} are fixed teaching assumptions. Energies in eV give the same numerical voltage in V in the last expression. At zero generated current, the displayed voltage and power are zero. The saturating yield is illustrative, not a universal offset threshold or a Marcus-rate calculation.')
        st.write('This effective-level model identifies the donor absorption peak with its gap and neglects exciton-binding/interfacial corrections. The donor HOMO and acceptor HOMO stay fixed; the donor LUMO follows its gap and the acceptor LUMO follows Δ. The acceptor gap is therefore derived, not an independent input. Acceptor absorption, transport, morphology, field-dependent collection and recombination-based voltage calculations are omitted. A Gaussian moves a finite absorption window; red-shifting does not always increase absorbed photon number.')
        st.markdown('Solar data: **ASTM G173 global-tilt AM1.5G**, distributed by [NLR/NREL](https://www.nlr.gov/grid/solar-resource/spectra-am1.5). The bundled table is the global column of pvlib’s reference-data distribution; no runtime download is needed.')
