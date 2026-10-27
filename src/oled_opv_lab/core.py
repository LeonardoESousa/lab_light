#!/usr/bin/env python3
"""Four-electron/four-orbital singlet--triplet gap model.

Spatial orbitals are ordered as

    1: lower occupied, theta_1
    2: HOMO, theta_2
    3: LUMO, theta_3
    4: LUMO+1, theta_4

and orbital 1 is the energy zero.  The user supplies epsilon_2, epsilon_3,
epsilon_4, the four angles, and a maximum exchange/coupling scale K_max.

The fragment amplitudes are u_p = (cos(theta_p), sin(theta_p)).  The effective
two-electron integral used for exchange and single--double coupling is

    g[p,q,r,s] = K_max * sum_X u[p,X] u[q,X] u[r,X] u[s,X], X in {D,A}.

It equals K_max when all four orbitals are completely localized on the same
fragment and vanishes for products on disjoint fragments.  No independent
Coulomb parameters are introduced.

The orbital energies are interpreted as canonical restricted-HF-like orbital
energies.  With direct Coulomb terms omitted, the standard spin-adapted CIS
diagonal elements are

    E0(S_HL)  = epsilon_3 - epsilon_2 + 2 K_23
    E0(T_HL)  = epsilon_3 - epsilon_2
    E0(S_HL1) = epsilon_4 - epsilon_2 + 2 K_24.

Thus the bare singlet--triplet splitting is 2 K_23.

All spin-adapted double excitations of the closed-shell (1)^2(2)^2 reference
within the four-orbital space are included.  For state X,

    delta_E_X^(2) = sum_D |<X|H2|D>|^2 / (E0_X - E0_D),

The double energies are not orbital sums alone.  For each spatial occupation
the code constructs the exchange-only Hamiltonian in its spin-adapted CSF
basis and diagonalizes it.  Consequently, singlet and triplet doubles have
different energies, and the two-singlet/three-triplet blocks of a four-open-
shell 2->2 configuration are treated explicitly.  The target couples only to
doubles of the same total spin, and its coupling vector is rotated into the
corresponding exchange eigenbasis.

Because the two singlets can be close, the code also constructs the Hermitian
second-order (van Vleck/Lowdin) 2x2 singlet effective Hamiltonian,

    H12^(2) = 1/2 sum_D V_1D V_D2
                    [1/(E1-E_D) + 1/(E2-E_D)],

and diagonalizes it.  The direct bare S_HL--S_HL1 coupling is taken to be zero
(the input orbitals are treated as canonical).  Both character-preserving and
adiabatic gaps are returned.

This is a qualitative effective model, not an ab-initio perturbation theory:
orbital relaxation, direct Coulomb integrals, and couplings to states outside
the four-orbital active space are neglected.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Dict, Mapping, Sequence

import numpy as np


N_ORBITALS = 4
OCCUPIED = (1, 2)
VIRTUAL = (3, 4)
HOMO = 2
SINGLE_CHOICES = tuple(
    (hole, particle) for hole in OCCUPIED for particle in VIRTUAL
)
State = Dict[int, float]
DoubleChoice = tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class ModelParameters:
    theta_1: float
    theta_lumo: float
    theta_lumo1: float
    epsilon_homo: float
    epsilon_lumo: float
    epsilon_lumo1: float
    exchange_scale: float
    cross_scale: float = 0.0
    theta_homo: float = 0.0


@dataclass
class StateEnergy:
    label: str
    spin: int
    bare_energy: float
    correction: float
    corrected_energy: float
    contributions: dict[str, float]
    coupling_squared: dict[str, float]
    denominators: dict[str, float]
    double_energies: dict[str, float]


def annihilate(det: int, q: int) -> tuple[int, int] | None:
    if not (det >> q) & 1:
        return None
    phase = -1 if (det & ((1 << q) - 1)).bit_count() % 2 else 1
    return det ^ (1 << q), phase


def create(det: int, p: int) -> tuple[int, int] | None:
    if (det >> p) & 1:
        return None
    phase = -1 if (det & ((1 << p) - 1)).bit_count() % 2 else 1
    return det | (1 << p), phase


def apply_string(
    det: int, creators: Sequence[int], annihilators: Sequence[int]
) -> tuple[int, int] | None:
    """Apply a^+_c0 ... a^+_cn a_am ... a_a0 to a determinant.

    ``creators=(c0,...,cn)`` and ``annihilators=(a0,...,am)`` therefore use
    the conventional two-body call ``(p,r), (q,s)`` for
    a^+_p a^+_r a_s a_q.  Operators act on the ket from right to left.
    """
    phase = 1
    for q in annihilators:
        result = annihilate(det, q)
        if result is None:
            return None
        det, local_phase = result
        phase *= local_phase
    for p in reversed(creators):
        result = create(det, p)
        if result is None:
            return None
        det, local_phase = result
        phase *= local_phase
    return det, phase


def scale_state(state: Mapping[int, float], factor: float) -> State:
    return {det: factor * coefficient for det, coefficient in state.items()}


def inner_product(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    return sum(coefficient * right.get(det, 0.0) for det, coefficient in left.items())


def reference_determinant() -> int:
    det = 0
    for orbital in OCCUPIED:
        det |= 1 << (2 * (orbital - 1))
        det |= 1 << (2 * (orbital - 1) + 1)
    return det


REFERENCE = reference_determinant()
DOUBLE_CHOICES: tuple[DoubleChoice, ...] = tuple(
    (holes, particles)
    for holes in itertools.combinations_with_replacement(OCCUPIED, 2)
    for particles in itertools.combinations_with_replacement(VIRTUAL, 2)
)


def single_state(particle: int, spin: int, hole: int = HOMO) -> State:
    """Normalized M_S=0 singlet or triplet for ``hole -> particle``."""
    components: list[State] = []
    for sigma in (0, 1):
        result = apply_string(
            REFERENCE,
            (2 * (particle - 1) + sigma,),
            (2 * (hole - 1) + sigma,),
        )
        assert result is not None
        det, phase = result
        components.append({det: float(phase)})

    # With the determinant ordering used here, + gives a singlet and - a triplet.
    relative_sign = 1.0 if spin == 0 else -1.0
    output: Dict[int, float] = defaultdict(float)
    for det, coefficient in components[0].items():
        output[det] += coefficient / math.sqrt(2.0)
    for det, coefficient in components[1].items():
        output[det] += relative_sign * coefficient / math.sqrt(2.0)
    return dict(output)


def spin_flip(state: Mapping[int, float], from_spin: int, to_spin: int) -> State:
    output: Dict[int, float] = defaultdict(float)
    for orbital in range(1, N_ORBITALS + 1):
        source = 2 * (orbital - 1) + from_spin
        target = 2 * (orbital - 1) + to_spin
        for det, coefficient in state.items():
            result = apply_string(det, (target,), (source,))
            if result is not None:
                new_det, phase = result
                output[new_det] += coefficient * phase
    return dict(output)


def s2_apply_ms0(state: Mapping[int, float]) -> State:
    """Apply S^2 in the M_S=0 sector."""
    minus_plus = spin_flip(spin_flip(state, 1, 0), 0, 1)
    plus_minus = spin_flip(spin_flip(state, 0, 1), 1, 0)
    output: Dict[int, float] = defaultdict(float)
    for det in set(minus_plus) | set(plus_minus):
        output[det] = 0.5 * (
            minus_plus.get(det, 0.0) + plus_minus.get(det, 0.0)
        )
    return dict(output)


def ms0_determinants() -> tuple[int, ...]:
    determinants = []
    for occupied_spin_orbitals in itertools.combinations(range(2 * N_ORBITALS), 4):
        n_alpha = sum(index % 2 == 0 for index in occupied_spin_orbitals)
        if n_alpha == 2:
            determinants.append(sum(1 << index for index in occupied_spin_orbitals))
    return tuple(determinants)


MS0_DETERMINANTS = ms0_determinants()


def excitation_signature(det: int) -> DoubleChoice:
    reference_occupations = (2, 2, 0, 0)
    occupations = tuple(
        ((det >> (2 * p)) & 1) + ((det >> (2 * p + 1)) & 1)
        for p in range(N_ORBITALS)
    )
    holes: list[int] = []
    particles: list[int] = []
    for orbital, (reference, current) in enumerate(
        zip(reference_occupations, occupations), start=1
    ):
        holes.extend([orbital] * max(0, reference - current))
        particles.extend([orbital] * max(0, current - reference))
    return tuple(holes), tuple(particles)  # type: ignore[return-value]


@lru_cache(maxsize=None)
def double_basis(choice: DoubleChoice, spin: int) -> tuple[State, ...]:
    """Return an orthonormal spin-adapted double-excitation basis."""
    determinants = tuple(
        det for det in MS0_DETERMINANTS if excitation_signature(det) == choice
    )
    s2 = np.zeros((len(determinants), len(determinants)))
    for column, det in enumerate(determinants):
        transformed = s2_apply_ms0({det: 1.0})
        for row, bra_det in enumerate(determinants):
            s2[row, column] = transformed.get(bra_det, 0.0)
    eigenvalues, eigenvectors = np.linalg.eigh(s2)
    target = spin * (spin + 1)
    basis: list[State] = []
    for index, eigenvalue in enumerate(eigenvalues):
        if math.isclose(float(eigenvalue), float(target), abs_tol=1e-9):
            basis.append(
                {
                    det: float(eigenvectors[row, index])
                    for row, det in enumerate(determinants)
                    if abs(eigenvectors[row, index]) > 1e-13
                }
            )
    return tuple(basis)


def h2_coefficient_tensor(bra: Mapping[int, float], ket: Mapping[int, float]) -> np.ndarray:
    """C[p,q,r,s] such that <bra|H2|ket> = sum C[p,q,r,s] g[p,q,r,s]."""
    coefficients = np.zeros((N_ORBITALS,) * 4, dtype=float)
    for p, q, r, s in itertools.product(range(1, N_ORBITALS + 1), repeat=4):
        value = 0.0
        for spin_pq, spin_rs in itertools.product((0, 1), repeat=2):
            creators = (2 * (p - 1) + spin_pq, 2 * (r - 1) + spin_rs)
            annihilators = (2 * (q - 1) + spin_pq, 2 * (s - 1) + spin_rs)
            for ket_det, ket_coefficient in ket.items():
                result = apply_string(ket_det, creators, annihilators)
                if result is None:
                    continue
                bra_det, phase = result
                value += 0.5 * bra.get(bra_det, 0.0) * ket_coefficient * phase
        coefficients[p - 1, q - 1, r - 1, s - 1] = value
    return coefficients


@lru_cache(maxsize=None)
def coupling_tensors(
    particle: int, spin: int, hole: int = HOMO
) -> dict[DoubleChoice, tuple[np.ndarray, ...]]:
    target = single_state(particle, spin, hole)
    return {
        choice: tuple(
            h2_coefficient_tensor(target, double_state)
            for double_state in double_basis(choice, spin)
        )
        for choice in DOUBLE_CHOICES
        if double_basis(choice, spin)
    }


def amplitudes(parameters: ModelParameters) -> np.ndarray:
    angles = (
        parameters.theta_1,
        parameters.theta_homo,
        parameters.theta_lumo,
        parameters.theta_lumo1,
    )
    return np.array([(math.cos(theta), math.sin(theta)) for theta in angles])


def interaction_tensor(parameters: ModelParameters) -> np.ndarray:
    u = amplitudes(parameters)
    pair_density = u[:, None, :] * u[None, :, :]
    metric = np.array(
        [[1.0, parameters.cross_scale], [parameters.cross_scale, 1.0]]
    )
    return parameters.exchange_scale * np.einsum(
        "pqx,xy,rsy->pqrs", pair_density, metric, pair_density, optimize=True
    )


def exchange_integral(g: np.ndarray, p: int, q: int) -> float:
    return float(g[p - 1, q - 1, q - 1, p - 1])


def exchange_only_tensor(g: np.ndarray) -> np.ndarray:
    """Retain K_pq=(pq|qp), excluding unphysical self-exchange terms."""
    exchange = np.zeros_like(g)
    for p in range(N_ORBITALS):
        for q in range(N_ORBITALS):
            if p != q:
                exchange[p, q, q, p] = g[p, q, q, p]
    return exchange


def orbital_energies(parameters: ModelParameters) -> np.ndarray:
    return np.array(
        [
            0.0,
            parameters.epsilon_homo,
            parameters.epsilon_lumo,
            parameters.epsilon_lumo1,
        ]
    )


def canonical_one_body_energies(energies: np.ndarray, exchange: np.ndarray) -> np.ndarray:
    """Recover h_pp from canonical epsilon_p when direct Coulomb is omitted.

    For the closed-shell reference, epsilon_p = h_pp - sum_i K_pi in this
    exchange-only model, where i runs over occupied spatial orbitals.
    """
    one_body = energies.copy()
    for p in range(N_ORBITALS):
        for occupied in (orbital - 1 for orbital in OCCUPIED):
            if p != occupied:
                one_body[p] += exchange[p, occupied, occupied, p]
    return one_body


def one_body_matrix_element(
    bra: Mapping[int, float], ket: Mapping[int, float], one_body: np.ndarray
) -> float:
    value = 0.0
    for det, ket_coefficient in ket.items():
        bra_coefficient = bra.get(det, 0.0)
        if bra_coefficient == 0.0:
            continue
        determinant_energy = sum(
            one_body[spin_orbital // 2]
            for spin_orbital in range(2 * N_ORBITALS)
            if (det >> spin_orbital) & 1
        )
        value += bra_coefficient * ket_coefficient * determinant_energy
    return float(value)


def hamiltonian_matrix_element(
    bra: Mapping[int, float],
    ket: Mapping[int, float],
    one_body: np.ndarray,
    exchange: np.ndarray,
) -> float:
    return one_body_matrix_element(bra, ket, one_body) + float(
        np.einsum(
            "pqrs,pqrs->",
            h2_coefficient_tensor(bra, ket),
            exchange,
            optimize=True,
        )
    )


@lru_cache(maxsize=None)
def double_h2_tensors(choice: DoubleChoice, spin: int) -> np.ndarray:
    """Cache two-electron coefficient tensors for a double-CSF block."""
    basis = double_basis(choice, spin)
    return np.array(
        [[h2_coefficient_tensor(bra, ket) for ket in basis] for bra in basis]
    )


def reference_energy(one_body: np.ndarray, exchange: np.ndarray) -> float:
    reference = {REFERENCE: 1.0}
    return hamiltonian_matrix_element(reference, reference, one_body, exchange)


