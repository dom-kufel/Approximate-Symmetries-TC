"""
Main entry point for the toric code simulation.

This script sets up and runs a variational Monte Carlo simulation of a toric code
model with various perturbations, using neural network quantum states as the variational
ansatz. It supports different architectural choices, and optimization
parameters. OBC are well tested, but PBC might require some additional work.

Author: D. Kufel
Last modified: March 24th, 2025
"""

import time
import uuid
import numpy as np
import netket as nk
import jax
import jax.numpy as jnp
import flax.linen as nn
import os
import sys
import json
from netket.utils import struct

from model.geometry import ToricCodeGeometry
from model.hamiltonian import create_hamiltonian
from model.networks import KernelManager, create_model
from simulation.optimizer import run_tdvp, create_final_callback
from simulation.observables import (
    create_wilson_loop_callback, create_magnetization_callback,
    create_renyi_callback, create_2point_callback, create_conditional_callbacks
)
from utils.config import setup_environment, parse_arguments, create_data_dict, save_data
from utils.io import save_model, log_runtime, record_experiment_info

# Import custom sampler if needed
from simulation.custom_sampler import create_custom_sampler

def main():
    # Start timing
    start_time = time.time()
    
    # Setup environment
    gpu_assigned, node_assigned, n_chains = setup_environment()
    
    # Parse command line arguments
    config = parse_arguments()
    
    # Override n_chains with device-specific value
    config['n_chains'] = n_chains
    
    # Print configuration
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create data dictionary
    data = create_data_dict(config, gpu_assigned, node_assigned)
    
    # Save initial data
    save_data(config['filename'], data)
    
    # Set up the geometry
    geometry = ToricCodeGeometry(config['Lx'], config['Ly'], config['bc'])
    
    # Create the Hilbert space
    hi = nk.hilbert.Spin(s=1/2, N=geometry.N)
    
    # Create the Hamiltonian
    H = create_hamiltonian(
        hi=hi,
        vertex_all=geometry.vertex_all,
        plaq_all=geometry.plaq_all,
        bonds=geometry.bonds,
        hx=config['hx'],
        hy=config['hy'],
        hz=config['hz'],
        J=config.get('J', 1.0),
        Jy_v=config.get('Jy_v', 0.0),
        Jy_p=config.get('Jy_p', 0.0),
        Jbond=config.get('Jbond', 0.0),
        dtype=config['dtype']
    )
    
    # Create the kernel manager
    kernel_manager = KernelManager(
        Lx=config['Lx'],
        Ly=config['Ly'],
        bc=config['bc'],
        kernel_size=config['kernel_size'],
        kernel_size_inv=config['kernel_size_inv'],
        arr_coord=geometry.arr_coord,
        dg_p=geometry.dg_p,
        N=geometry.N
    )
    
    # Create the neural network model
    model = create_model(config, geometry.plaq_all, kernel_manager)
    print(model)
    
    # Create a sampler based on configuration
    if config.get('use_custom_sampler', False):
        # Use custom sampler with vertex updates
        sa = create_custom_sampler(geometry, hi, config)
        print("Using custom sampler with vertex updates")
    else:
        # Use standard sampler with local rule
        rule = nk.sampler.rules.LocalRule()
        sa = nk.sampler.MetropolisSampler(
            hi, 
            rule=rule, 
            n_chains=config['n_chains'],
            n_sweeps=config['n_sweeps'],
            dtype=jnp.int8
        )
        print("Using standard sampler with local updates")
    
    # Create the variational state
    vs = nk.vqs.MCState(
        sa, 
        model, 
        n_samples=config['n_samples'],
        n_discard_per_chain=config['n_discard'],
        chunk_size=config['chunk_size']
    )
    
    # Update number of parameters in the data dictionary
    with open(config['filename'], 'r') as f:
        data = json.load(f)
    data["sim_params"]["n_params"] = [vs.n_parameters]
    with open(config['filename'], 'w') as f:
        json.dump(data, f)
    
    # Setup callbacks for observables
    callbacks = create_conditional_callbacks(geometry)
    
    # Print information before starting optimization
    print(f"Number of qubits: {geometry.N}")
    print(f"Number of model parameters: {vs.n_parameters}")
    print(f"Starting optimization...")
    
    # Run the optimization
    vs = run_tdvp(
        hamiltonian=H,
        vstate=vs,
        config=config,
        callbacks=callbacks
    )
    
    # Save the final model
    save_model(vs, config['filename_base'])
    
    # Calculate observables
    print("Calculating final observables...")
    
    # For Lx > 6, calculate all observables at the end
    if geometry.Lx > 6:
        # Calculate Wilson loops
        callback = create_wilson_loop_callback(geometry)
        callback(vs, -1, -1, config)
        
        # Calculate Renyi entropy
        callback = create_renyi_callback(geometry)
        callback(vs, -1, -1, config)
        
        # # Calculate two-point correlation functions
        # callback = create_2point_callback(geometry) #doesn't work yet
        # callback(vs, -1, -1, config)

    # Always calculate magnetizations at the end
    callback = create_magnetization_callback(geometry)
    callback(vs, -1, -1, config)
    
    # Log runtime
    log_runtime(config, start_time)
    
    # Record experiment information with a unique ID
    run_id = str(uuid.uuid4())[:8]
    record_experiment_info(
        config=config,
        run_id=run_id,
        description="Toric code simulation with neural network quantum states",
        extra_info={
            "n_params": vs.n_parameters,
            "final_energy": vs.expect(H).mean,
            "runtime": time.time() - start_time
        }
    )
    
    print("Simulation complete.")

if __name__ == "__main__":
    main() 