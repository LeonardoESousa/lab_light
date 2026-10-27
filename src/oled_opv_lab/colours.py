"""NEMOview-style spectrum → CIE XYZ → sRGB and proportional vial rendering.

Adapted from the installed nemoview.visualization.FluorescentVialPlotter.
Preserves its D65-weighted conversion for continuity; normalizes XYZ and
linear RGB before encoding to avoid the original all-white clipping.
An emission-only conversion is available with legacy_d65=False.
"""
import numpy as np
import colour


def spectrum_to_color(wavelengths_nm, intensities, legacy_d65=True):
    w = np.asarray(wavelengths_nm, float)
    y = np.asarray(intensities, float)
    if w.ndim != 1 or w.shape != y.shape or len(w) < 2:
        raise ValueError('Supply matching one-dimensional wavelength and intensity arrays.')
    if np.any(~np.isfinite(w)) or np.any(~np.isfinite(y)) or np.any(y < 0):
        raise ValueError('Spectrum must be finite and nonnegative.')
    idx = np.argsort(w)
    w, y = w[idx], y[idx]
    if np.any(np.diff(w) <= 0):
        raise ValueError('Wavelengths must be unique.')
    grid = np.arange(380., 781.)
    visible = np.interp(grid, w, y, left=0., right=0.)
    # Do not brighten an effectively invisible UV/IR tail into a vivid vial.
    if y.max() <= 0 or visible.max() < 1e-6*y.max():
        return (24, 28, 38)
    shape = colour.SpectralShape(380,780,1)
    sd = colour.SpectralDistribution(dict(zip(grid, visible/visible.max())))
    cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].copy().align(shape)
    illuminant = colour.SDS_ILLUMINANTS['D65' if legacy_d65 else 'E'].copy().align(shape)
    xyz = colour.sd_to_XYZ(sd, cmfs=cmfs, illuminant=illuminant, method='Integration')/100
    linear = np.maximum(colour.XYZ_to_sRGB(xyz, apply_cctf_encoding=False), 0)
    if linear.max() > 0:
        linear /= linear.max()
    rgb = np.clip(colour.cctf_encoding(linear, function='sRGB'), 0, 1)
    return tuple(int(round(x*255)) for x in rgb)


def scale_emission_color(rgb, strength):
    """Scale linear-light display brightness without renormalizing it away."""
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Relative emission strength must lie in [0, 1].')
    srgb = np.asarray(rgb, float)/255
    linear = colour.cctf_decoding(srgb, function='sRGB')
    scaled = colour.cctf_encoding(linear*strength, function='sRGB')
    return tuple(int(round(v*255)) for v in np.clip(scaled,0,1))


def vial_svg(rgb):
    """Fixed viewBox keeps cap, liquid and outline proportional at every size."""
    r, g, b = (int(np.clip(c, 0, 255)) for c in rgb)
    color = f'rgb({r},{g},{b})'
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260 320" role="img" aria-label="Emission colour vial">
    <defs><radialGradient id="halo"><stop stop-color="{color}" stop-opacity=".32"/><stop offset="1" stop-color="{color}" stop-opacity="0"/></radialGradient>
    <linearGradient id="glass"><stop stop-color="white" stop-opacity=".6"/><stop offset=".35" stop-color="white" stop-opacity="0"/><stop offset="1" stop-color="white" stop-opacity=".2"/></linearGradient></defs>
    <rect width="260" height="320" rx="18" fill="#101925"/>
    <ellipse cx="130" cy="190" rx="115" ry="120" fill="url(#halo)"/>
    <rect x="80" y="105" width="100" height="170" rx="14" fill="{color}"/>
    <ellipse cx="130" cy="108" rx="50" ry="8" fill="{color}" stroke="white" stroke-opacity=".4"/>
    <rect x="75" y="64" width="110" height="216" rx="17" fill="url(#glass)" stroke="#d8e4ef" stroke-width="3"/>
    <rect x="73" y="42" width="114" height="35" rx="6" fill="#47586b" stroke="#b5c5d5" stroke-width="2"/>
    <path d="M90 49V69 M103 49V69 M116 49V69 M129 49V69 M142 49V69 M155 49V69 M168 49V69" stroke="#8d9dad" stroke-width="3"/>
    <path d="M89 90V251" stroke="white" stroke-opacity=".4" stroke-width="5" stroke-linecap="round"/>
    </svg>'''
