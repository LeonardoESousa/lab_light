"""Schematic donor/acceptor density profiles, not real-space orbital grids."""
import numpy as np
import plotly.graph_objects as go


def orbital_density_figure(angles, labels=None):
    """Fixed D/A halves, equal-width lobes; their areas scale with populations.

    Labels and angles are supplied from lowest to highest orbital. The default
    order is H−1,H,L,L+1. For two orbitals use labels=('HOMO','LUMO').
    """
    angles=np.asarray(angles,dtype=float)
    labels=tuple(labels) if labels is not None else ('H−1','H','L','L+1')
    if not labels or angles.shape!=(len(labels),) or np.any(~np.isfinite(angles)) or np.any((angles<0)|(angles>90)):
        raise ValueError('Supply one finite angle between 0 and 90 degrees per orbital label.')
    donor=np.cos(np.radians(angles))**2
    fig=go.Figure()
    for row,(label,pd) in enumerate(zip(labels,donor)):
        bottom,top=row-.30,row+.30
        for x0,x1,color in [(-1,0,'#e5f3f4'),(0,1,'#fff0dc')]:
            fig.add_shape(type='rect',x0=x0,x1=x1,y0=bottom,y1=top,
                          fillcolor=color,line=dict(color='#b9c7d4',width=1),layer='below')
        fig.add_shape(type='line',x0=0,x1=0,y0=bottom,y1=top,
                      line=dict(color='#889bad',width=1,dash='dot'))
        for x0,weight,color,fragment in [(-1,float(pd),'#087e8b','D'),(0,float(1-pd),'#d88932','A')]:
            x=np.linspace(x0,x0+1,101)
            # Identical envelopes on both fragments keep lobe-area ratios exact.
            envelope=np.sin(np.pi*(x-x0))**2
            y=bottom+.56*weight*envelope
            fig.add_scatter(x=np.r_[x,x[::-1]],y=np.r_[y,np.full_like(x,bottom)],
                            mode='lines',fill='toself',fillcolor=color,
                            line=dict(color=color,width=1.5),showlegend=False,
                            name=f'{label} · {fragment}',
                            hovertemplate=f'{label} · {fragment}: {100*weight:.1f}%<extra></extra>')
    fig.update_xaxes(range=[-1.05,1.05],tickvals=[-.5,.5],
                     ticktext=['Donor (D)','Acceptor (A)'],showgrid=False,zeroline=False)
    fig.update_yaxes(range=[-.55,len(labels)-.45],tickvals=list(range(len(labels))),
                     ticktext=list(labels),showgrid=False,zeroline=False,
                     fixedrange=True)
    fig.update_layout(showlegend=False)
    return fig
