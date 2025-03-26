"""
Module for creating the toric code Hamiltonian with various perturbations.
"""

import netket as nk
import numpy as np
import jax.numpy as jnp
from typing import List, Dict, Any, Tuple, Optional, Union

def create_hamiltonian(
    hi: nk.hilbert.Spin,
    vertex_all: List[List[int]],
    plaq_all: List[List[int]],
    bonds: List[List[int]],
    hx: float = 0.0,
    hy: float = 0.0,
    hz: float = 0.0,
    J: float = 1.0,
    Jy_v: float = 0.0,
    Jy_p: float = 0.0,
    Jbond: float = 0.0,
    dtype: Any = complex
) -> nk.operator.AbstractOperator:
    """
    Create the toric code Hamiltonian with perturbations.
    
    Args:
        hi: Hilbert space
        vertex_all: List of vertex operators
        plaq_all: List of plaquette operators
        bonds: List of nearest-neighbor bonds
        hx: X magnetic field strength
        hy: Y magnetic field strength
        hz: Z magnetic field strength
        J: Coupling strength
        Jy_v: Y vertex coupling
        Jy_p: Y plaquette coupling
        Jbond: Bond coupling
        dtype: Data type for the Hamiltonian
        
    Returns:
        The toric code Hamiltonian
    """
    H = 0
    N = hi.size
    
    # Add vertex terms
    for v in range(0, len(vertex_all)):
        # XXXX vertex terms
        op = 1
        for j in range(0, len(vertex_all[v])):
            if vertex_all[v][j] != -1:
                op *= nk.operator.spin.sigmax(hi, vertex_all[v][j], dtype=dtype)
        H += -J * op
        
        # YYYY vertex terms
        if Jy_v != 0:
            assert dtype == "complex", "YYYY vertex terms require complex Hamiltonian"
            op = 1
            for j in range(0, len(vertex_all[v])):
                if vertex_all[v][j] != -1:
                    op *= nk.operator.spin.sigmay(hi, vertex_all[v][j], dtype=dtype)
            H += -Jy_v * op
    
    # Add plaquette terms
    for p in range(0, len(plaq_all)):
        # ZZZZ plaquette terms
        op = 1
        for j in range(0, len(plaq_all[p])):
            if plaq_all[p][j] != -1:
                op *= nk.operator.spin.sigmaz(hi, plaq_all[p][j], dtype=dtype)
        H += -J * op
        
        # YYYY plaquette terms
        if Jy_p != 0:
            assert dtype == "complex", "YYYY plaquette terms require complex Hamiltonian"
            op = 1
            for j in range(0, len(plaq_all[p])):
                if plaq_all[p][j] != -1:
                    op *= nk.operator.spin.sigmay(hi, plaq_all[p][j], dtype=dtype)
            H += -Jy_p * op
    
    # Add magnetic field perturbations
    for j in range(0, N):
        if hz != 0:
            H += -nk.operator.spin.sigmaz(hi, j, dtype=dtype) * hz
        if hx != 0:
            H += -nk.operator.spin.sigmax(hi, j, dtype=dtype) * hx
        if hy != 0:
            assert dtype == "complex", "Y magnetic field requires complex Hamiltonian"
            H += -nk.operator.spin.sigmay(hi, j, dtype=dtype) * hy
    
    # Add 2-qubit perturbations (bonds)
    if Jbond != 0.0:
        for (x, y) in bonds:
            H += -nk.operator.spin.sigmax(hi, x) * nk.operator.spin.sigmax(hi, y) * Jbond
            H += -nk.operator.spin.sigmaz(hi, x) * nk.operator.spin.sigmaz(hi, y) * Jbond
            H += -nk.operator.spin.sigmay(hi, x) * nk.operator.spin.sigmay(hi, y) * Jbond
    
    # Convert to Pauli strings for more efficient implementation
    H = H.to_pauli_strings()
    
    return H 