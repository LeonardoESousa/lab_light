#!/usr/bin/env python3
"""Spin-adapted CISD angle scan for the four-electron/four-orbital model.

Orbital 2 (HOMO) is fixed on D and orbital 3 (LUMO) on A.  The localization
angles of orbital 1 (HOMO-1) and orbital 4 (LUMO+1) are scanned from D=0 to
A=pi/2.  Separate singlet and triplet CISD Hamiltonians are diagonalized.

The CISD spaces contain all four single excitations and every double excitation
from the closed-shell (1)^2(2)^2 reference:

    singlet: reference + 4 singles + 10 double CSFs = 15 CSFs
    triplet:             4 singles +  7 double CSFs = 11 CSFs

Diagonal blocks use canonical orbital energies plus exchange terms only.  The
off-diagonal correlation matrix elements use the same localization-dependent
four-index interaction proxy as four_orbital_state_gaps.py.  Reference--single
matrix elements are set to zero (canonical-orbital Brillouin condition).
Direct Coulomb contributions to configuration energies are omitted.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np

from . import core


Signature = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class CSF:
    label: str
    signature: Signature
    state: core.State

    @property
    def rank(self) -> int:
        return len(self.signature[0])


def signature_for_determinant(det: int) -> Signature:
    reference_occupations = (2, 2, 0, 0)
    occupations = tuple(
        ((det >> (2 * p)) & 1) + ((det >> (2 * p + 1)) & 1)
        for p in range(core.N_ORBITALS)
    )
    holes: list[int] = []
    particles: list[int] = []
    for orbital, (reference, current) in enumerate(
        zip(reference_occupations, occupations), start=1
    ):
        holes.extend([orbital] * max(0, reference - current))
        particles.extend([orbital] * max(0, current - reference))
    return tuple(holes), tuple(particles)


@lru_cache(maxsize=None)
def spin_basis(signature: Signature, spin: int) -> tuple[core.State, ...]:
    determinants = tuple(
        det
        for det in core.MS0_DETERMINANTS
        if signature_for_determinant(det) == signature
    )
    s2 = np.zeros((len(determinants), len(determinants)))
    for column, det in enumerate(determinants):
        transformed = core.s2_apply_ms0({det: 1.0})
        for row, bra_det in enumerate(determinants):
            s2[row, column] = transformed.get(bra_det, 0.0)
    eigenvalues, eigenvectors = np.linalg.eigh(s2)
    target = spin * (spin + 1)
    output: list[core.State] = []
    for column, eigenvalue in enumerate(eigenvalues):
        if math.isclose(float(eigenvalue), float(target), abs_tol=1e-9):
            output.append(
                {
                    det: float(eigenvectors[row, column])
                    for row, det in enumerate(determinants)
                    if abs(eigenvectors[row, column]) > 1e-13
                }
            )
    return tuple(output)


def cisd_basis(spin: int) -> tuple[CSF, ...]:
    signatures: list[Signature] = []
    if spin == 0:
        signatures.append(((), ()))
    signatures.extend(
        ((hole,), (particle,))
        for hole in core.OCCUPIED
        for particle in core.VIRTUAL
    )
    signatures.extend(core.DOUBLE_CHOICES)

    basis: list[CSF] = []
    for signature in signatures:
        states = spin_basis(signature, spin)
        for index, state in enumerate(states, start=1):
            holes, particles = signature
            if not holes:
                label = "reference"
            else:
                left = ",".join(map(str, holes))
                right = ",".join(map(str, particles))
                label = f"({left})->({right})[{('S' if spin == 0 else 'T')}{index}]"
            basis.append(CSF(label, signature, state))
    return tuple(basis)


def number_coefficient(
    bra: Mapping[int, float], ket: Mapping[int, float]
) -> np.ndarray:
    coefficients = np.zeros(core.N_ORBITALS)
    for det, ket_coefficient in ket.items():
        bra_coefficient = bra.get(det, 0.0)
        if bra_coefficient == 0.0:
            continue
        for spatial in range(core.N_ORBITALS):
            occupation = ((det >> (2 * spatial)) & 1) + (
                (det >> (2 * spatial + 1)) & 1
            )
            coefficients[spatial] += (
                bra_coefficient * ket_coefficient * occupation
            )
    return coefficients


@dataclass(frozen=True)
class PreparedSpace:
    spin: int
    basis: tuple[CSF, ...]
    number_tensors: np.ndarray
    interaction_tensors: np.ndarray
    same_signature: np.ndarray
    reference_single: np.ndarray


def prepare_space(spin: int) -> PreparedSpace:
    basis = cisd_basis(spin)
    size = len(basis)
    numbers = np.zeros((size, size, core.N_ORBITALS))
    interactions = np.zeros((size, size) + (core.N_ORBITALS,) * 4)
    same_signature = np.zeros((size, size), dtype=bool)
    reference_single = np.zeros((size, size), dtype=bool)
    for i, bra in enumerate(basis):
        for j, ket in enumerate(basis):
            numbers[i, j] = number_coefficient(bra.state, ket.state)
            interactions[i, j] = core.h2_coefficient_tensor(
                bra.state, ket.state
            )
            same_signature[i, j] = bra.signature == ket.signature
            reference_single[i, j] = {bra.rank, ket.rank} == {0, 1}
    return PreparedSpace(
        spin, basis, numbers, interactions, same_signature, reference_single
    )


SINGLET_SPACE = prepare_space(0)
TRIPLET_SPACE = prepare_space(1)


def build_hamiltonian(
    space: PreparedSpace,
    one_body: np.ndarray,
    exchange: np.ndarray,
    coupling_integrals: np.ndarray,
    reference_energy: float,
) -> np.ndarray:
    h_base = np.einsum("ijp,p->ij", space.number_tensors, one_body)
    h_base += np.einsum(
        "ijpqrs,pqrs->ij", space.interaction_tensors, exchange, optimize=True
    )
    h_coupling = np.einsum(
        "ijpqrs,pqrs->ij",
        space.interaction_tensors,
        coupling_integrals,
        optimize=True,
    )
    hamiltonian = np.where(space.same_signature, h_base, h_coupling)
    hamiltonian[space.reference_single] = 0.0
    hamiltonian -= reference_energy * np.eye(len(space.basis))
    return 0.5 * (hamiltonian + hamiltonian.T)


