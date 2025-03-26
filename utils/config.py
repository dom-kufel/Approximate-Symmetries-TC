"""
Configuration utilities for handling command line arguments and simulation parameters.
"""

import argparse
import json
import os
from typing import Dict, Any, List, Union, Optional
import time
import subprocess
import sys
import platform
import jax

def setup_environment():
    """Configure environment variables and print hardware information."""
    # Check for GPU availability using jax.devices()
    devices = jax.devices()
    has_gpu = any('cuda' in str(device).lower() for device in devices)
    
    if has_gpu:
        # GPU mode
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
        gpu_assigned = "NVIDIA GPU"
        n_chains = 2**10  # 1024 chains for GPU
        
        # Try to get GPU information
        try:
            command = 'nvidia-smi --query-gpu=gpu_name --format=csv,noheader'
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            gpu_name, error = process.communicate()
            gpu_assigned = str(gpu_name)
        except FileNotFoundError:
            pass
    else:
        # CPU mode
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        gpu_assigned = "CPU mode"
        n_chains = 2**4  # 16 chains for CPU
    
    # Get hostname
    try:
        command2 = 'hostname'
        process2 = subprocess.Popen(command2.split(), stdout=subprocess.PIPE)
        node_assigned, error = process2.communicate()
        node_assigned = str(node_assigned)
    except FileNotFoundError:
        node_assigned = platform.node()

    print("NODE:", node_assigned)
    print("ASSIGNED DEVICE:", gpu_assigned)
    print("NUMBER OF CHAINS:", n_chains)
    print("AVAILABLE DEVICES:", devices)
    
    return gpu_assigned, node_assigned, n_chains

def parse_arguments() -> Dict[str, Any]:
    """Parse command line arguments and return as a dictionary.
    
    Returns:
        dict: Dictionary containing all configuration parameters
    """
    parser = argparse.ArgumentParser(description="Quantum Many-Body Simulation with JAX and NetKet")
    
    # Output file parameters
    parser.add_argument('--outindex', required=True, help='Output index for filenames')
    parser.add_argument('--jobid', required=True, help='Job ID for filenames')
    parser.add_argument('--annotation', type=str, default="cluster_16x16_run_hy", 
                       help='Annotation for the output files')
    
    # Geometry parameters
    parser.add_argument('--Lx', type=int, default=2, help='Number of vertices in x direction')
    parser.add_argument('--bc', type=str, choices=['OBC', 'PBC'], default='OBC', help='Boundary conditions')
    
    # Hamiltonian parameters
    parser.add_argument('--hx', type=float, required=True, help='X magnetic field strength')
    parser.add_argument('--hy', type=float, required=True, help='Y magnetic field strength')
    parser.add_argument('--hz', type=float, required=True, help='Z magnetic field strength')
    parser.add_argument('--J', type=float, default=1.0, help='Coupling strength')
    parser.add_argument('--Jy_p', type=float, default=0.0, help='Y plaquette coupling')
    parser.add_argument('--Jy_v', type=float, default=0.0, help='Y vertex coupling')
    parser.add_argument('--Jbond', type=float, default=0.0, help='Bond coupling')
    
    # Optimization parameters
    parser.add_argument('--dt', type=float, required=True, help='Time step')
    parser.add_argument('--diag_shift', type=float, required=True, help='Diagonal shift')
    parser.add_argument('--sim_time', type=float, default=3.5, help='Simulation time')
    
    # Neural network parameters
    parser.add_argument('--architecture', type=str, choices=['Combo', 'RPP'], default='Combo', 
                        help='Architecture type')
    parser.add_argument('--channels_noninv', type=str, required=True, 
                        help='Comma-separated integers for non-invariant channels')
    parser.add_argument('--channels_inv', type=str, required=True,
                        help='Comma-separated integers for invariant channels')
    parser.add_argument('--kernel_size', type=int, required=True, help='Kernel size for non-invariant CNN')
    parser.add_argument('--rescale', type=float, default=1.0, help='Rescale factor')
    
    # MCMC sampling parameters
    parser.add_argument('--n_samples', type=int, default=2**13, help='Total number of samples')
    parser.add_argument('--n_chains', type=int, help='Number of MCMC chains (will be overridden by device detection)')
    parser.add_argument('--n_discard', type=int, default=2**3, help='Number of burn-in steps per chain')
    parser.add_argument('--chunk_size', type=int, default=2**11, help='Max number of samples to process in parallel')
    parser.add_argument('--n_sweeps', type=int, help='Number of subsampling steps (defaults to N/2)')
    parser.add_argument('--n_samples_fin', type=int, required=True, help='Final number of samples')
    parser.add_argument('--use_custom_sampler', action='store_true', help='Use custom sampler with vertex updates')
    
    # Parse arguments
    if len(sys.argv) <= 12:  # Check if using old positional arguments format
        # For backward compatibility, read from sys.argv directly
        args = {
            'outindex': sys.argv[1],
            'jobid': sys.argv[2],
            'hx': float(eval(sys.argv[3])),
            'hy': float(eval(sys.argv[4])),
            'hz': float(eval(sys.argv[5])),
            'dt': float(eval(sys.argv[6])),
            'diag_shift': float(eval(sys.argv[7])),
            'channels_noninv': [int(el) for el in sys.argv[8].split(',')],
            'channels_inv': [int(el) for el in sys.argv[9].split(',')],
            'kernel_size': int(eval(sys.argv[10])),
            'n_samples_fin': int(eval(sys.argv[11])),
            'architecture': 'Combo',  # Default values for backwards compatibility
            'bc': 'OBC',
            'J': 1.0,
            'Jy_p': 0.0,
            'Jy_v': 0.0,
            'Jbond': 0.0,
            'n_samples': 2**13,
            'n_chains': 2**10,  # Will be overridden by device detection
            'n_discard': 2**3,
            'chunk_size': 2**11,
            'n_sweeps': 2**10 // 2,  # Will be overridden by N/2 if not provided
            'sim_time': 3.5,
            'rescale': 1.0,
            'annotation': "cluster_16x16_run_hy"
        }
    else:
        args = vars(parser.parse_args())
        # Convert channel strings to lists of integers
        args['channels_noninv'] = [int(el) for el in args['channels_noninv'].split(',')]
        args['channels_inv'] = [int(el) for el in args['channels_inv'].split(',')]
    
    # Set Ly equal to Lx
    args['Ly'] = args['Lx']
    
    # Determine N based on boundary conditions
    if args['bc'] == "OBC":
        args['N'] = 2 * args['Lx'] * (args['Lx'] - 1)
    else:
        args['N'] = 2 * args['Lx'] * args['Ly']
    
    # Set default n_sweeps to N/2 if not provided
    if args.get('n_sweeps') is None:
        args['n_sweeps'] = args['N'] // 2
    
    # Calculate kernel_size_inv based on Lx
    args['kernel_size_inv'] = args['Lx'] - 1
    
    # Determine dtype based on parameters
    if args['hy'] != 0.0 or args['Jy_p'] != 0.0 or args['Jy_v'] != 0.0:
        args['dtype'] = "complex"
    else:
        args['dtype'] = "float64"
    
    # Set filenames
    args['filename_base'] = f"G-equiv_{args['outindex']}_{args['jobid']}"
    args['filename'] = f"{args['filename_base']}.json"
    args['filename_mpack'] = f"{args['filename_base']}.mpack"
    
    return args

def create_data_dict(config: Dict[str, Any], gpu_assigned: str, node_assigned: str) -> Dict[str, Any]:
    """Create the initial data dictionary for storing simulation results.
    
    Args:
        config: Configuration dictionary
        gpu_assigned: GPU assigned to the job
        node_assigned: Compute node assigned to the job
        
    Returns:
        dict: Data dictionary for storing simulation results
    """
    return {
        "iters": [],
        "energy": [],
        "energy_eom": [],
        "energy_var": [],
        "tau_corr": [],
        "Rsplit": [],
        "Vscore": [],
        "MCMC_accepted": [],
        "MCMC_total": [],
        "equiv_error": [],
        "equiv_error_bulk": [],
        "order_params": {
            "magnetization_Xmean": [], 
            "magnetization_Xstd": [],
            "magnetization_Ymean": [], 
            "magnetization_Ystd": [],
            "magnetization_Zmean": [], 
            "magnetization_Zstd": [],
            "2pointCorrelators": [],
            "WilsonBFFM": [],
            "renyi2_entropy": []
        },
        "sim_params": {
            "kind": ["G-NonInv"],
            "architecture_type": [config["architecture"]],
            "Lx": [config["Lx"]],
            "hx": [config["hx"]],
            "hy": [config["hy"]],
            "hz": [config["hz"]],
            "Jy_p": [config["Jy_p"]],
            "Jy_v": [config["Jy_v"]],
            "Jbond": [config["Jbond"]],
            "BC": [config["bc"]],
            "n_chann_inv": config["channels_inv"],
            "n_chann_noninv": config["channels_noninv"],
            "rescale": [config["rescale"]],
            "kernel_size_noninv": [config["kernel_size"]],
            "kernel_size_inv": [config["kernel_size_inv"]],
            "n_params": [0],  # Will be updated later
            "n_samples": [config["n_samples"]],
            "n_samples_fin": [config["n_samples_fin"]],
            "n_sweeps_fin": ["None"],
            "n_chains": [config["n_chains"]],
            "n_discard_per_chain=": [config["n_discard"]],
            "n_sweeps": [config["n_sweeps"]],
            "chunk_size=": [config["chunk_size"]],
            "dt": [config["dt"]],
            "diag_shift": [config["diag_shift"]],
            "diag_shift_init": ["None"],
            "param_dtype": [str(config["dtype"])],
            "runtime": [],
            "annotation": [config["annotation"]],
            "gpu:assigned": [gpu_assigned],
            "node:assigned": [node_assigned]
        }
    }

def save_data(filename: str, data: Dict[str, Any]) -> None:
    """Save data to a JSON file.
    
    Args:
        filename: Name of the JSON file
        data: Data dictionary to save
    """
    with open(filename, 'w') as f:
        json.dump(data, f)

def update_data(filename: str, keys: List[str], values: List[Any]) -> None:
    """Update data in a JSON file.
    
    Args:
        filename: Name of the JSON file
        keys: List of keys to update
        values: List of values to add to the corresponding keys
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    for i, key in enumerate(keys):
        data[key].append(str(values[i]))
    
    with open(filename, 'w') as f:
        json.dump(data, f) 