"""Full diagonalization of the established effective four-orbital CISD model."""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from . import core
from .cisd_basis import SINGLET_SPACE, TRIPLET_SPACE, build_hamiltonian


@dataclass(frozen=True)
class OrbitalParameters:
    angles: tuple = (0., 0., 90., 0.)
    energies: tuple = (-0.2528, -0.2527, 2.7733, 3.3492)
    k_max: float = 0.1
    cross_scale: float = 0.

    def validate(self):
        if len(self.angles) != 4 or len(self.energies) != 4:
            raise ValueError("Exactly four angles and orbital energies are required.")
        if not np.all(np.isfinite([*self.angles, *self.energies, self.k_max, self.cross_scale])):
            raise ValueError("All inputs must be finite.")
        if any(not 0 <= a <= 90 for a in self.angles):
            raise ValueError("Angles must lie between 0 and 90 degrees.")
        if not np.all(np.diff(self.energies) > 0):
            raise ValueError("Require H−1 < H < L < L+1.")
        if self.k_max < 0 or not 0 <= self.cross_scale <= 1:
            raise ValueError("Require Kmax ≥ 0 and 0 ≤ λ ≤ 1.")


@lru_cache(maxsize=512)
def solve(parameters=OrbitalParameters()):
    parameters.validate()
    a = np.radians(parameters.angles)
    e = np.array(parameters.energies) - parameters.energies[0]
    p = core.ModelParameters(a[0], a[2], a[3], e[1], e[2], e[3],
                             parameters.k_max, parameters.cross_scale, a[1])
    g = core.interaction_tensor(p)
    exchange = core.exchange_only_tensor(g)
    h = core.canonical_one_body_energies(e, exchange)
    ref = core.reference_energy(h, exchange)
    hs = build_hamiltonian(SINGLET_SPACE, h, exchange, g, ref)
    ht = build_hamiltonian(TRIPLET_SPACE, h, exchange, g, ref)
    es, vs = np.linalg.eigh(hs)
    et, vt = np.linalg.eigh(ht)
    single_s = np.array([c.rank == 1 for c in SINGLET_SPACE.basis])
    single_t = np.array([c.rank == 1 for c in TRIPLET_SPACE.basis])
    hl_s = next(i for i, c in enumerate(SINGLET_SPACE.basis)
                if c.signature == ((2,), (3,)))
    hl_t = next(i for i, c in enumerate(TRIPLET_SPACE.basis)
                if c.signature == ((2,), (3,)))
    gap = float(es[1]-et[0])
    if abs(gap) < 1e-12:
        gap = 0.
    return dict(s0=float(es[0]), s1=float(es[1]-es[0]), s2=float(es[2]-es[0]),
                t1=float(et[0]-es[0]), gap=gap,
                singlet_energies=es-es[0], triplet_energies=et-es[0],
                singlet_vectors=vs, triplet_vectors=vt,
                s1_single=float(np.sum(vs[single_s, 1]**2)),
                t1_single=float(np.sum(vt[single_t, 0]**2)),
                s1_hl=float(vs[hl_s, 1]**2),
                t1_hl=float(vt[hl_t, 0]**2),
                s0_reference=float(vs[0, 0]**2), hs=hs, ht=ht)


def eligible(result, minimum_hl=.8):
    """Only reward gaps of the lowest roots if BOTH remain H→L dominated."""
    return (result['s1'] > 0 and result['t1'] > 0
            and result['s1_hl'] >= minimum_hl
            and result['t1_hl'] >= minimum_hl)


def minimum_frontier_spacing(k_max):
    """Conservative bare-singles isolation; verify correlated character separately."""
    if not np.isfinite(k_max) or k_max < 0:
        raise ValueError('Kmax must be finite and nonnegative.')
    return round(2 * k_max + .05, 4)


def composition(result, state):
    space, vector = ((TRIPLET_SPACE, result['triplet_vectors'][:, 0]) if state == 'T1'
                     else (SINGLET_SPACE, result['singlet_vectors'][:, {'S0':0,'S1':1,'S2':2}[state]]))
    labels = {1:'H−1',2:'H',3:'L',4:'L+1'}
    rows = []
    for csf, weight in zip(space.basis, vector**2):
        holes, particles = csf.signature
        name = ('Reference' if not holes else
                ','.join(labels[x] for x in holes)+' → '+','.join(labels[x] for x in particles))
        rows.append({'Configuration':name, 'Rank':csf.rank, 'Weight (%)':float(100*weight)})
    return rows
