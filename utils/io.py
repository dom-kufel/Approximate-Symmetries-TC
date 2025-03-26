"""
Module for handling I/O operations like saving and loading models and data.
"""

import json
import time
import flax
import flax.linen as nn
import netket as nk
from typing import Dict, Any, Optional, List

def save_model(vstate: nk.vqs.VariationalState, filename: str) -> None:
    """
    Save a variational state to a file.
    
    Args:
        vstate: NetKet variational state
        filename: Name of the file to save to (without extension)
    """
    with open(f"{filename}.mpack", 'wb') as file:
        file.write(flax.serialization.to_bytes(vstate))
    
    print(f"Model saved to {filename}.mpack")

def load_model(filename: str, sampler: nk.sampler.MetropolisSampler, model: nn.Module, 
               n_samples: int, n_discard_per_chain: int, chunk_size: int) -> nk.vqs.MCState:
    """
    Load a variational state from a file.
    
    Args:
        filename: Name of the file to load from (without extension)
        sampler: NetKet sampler
        model: Flax neural network model
        n_samples: Number of samples
        n_discard_per_chain: Number of burn-in steps per chain
        chunk_size: Chunk size
        
    Returns:
        Loaded variational state
    """
    # Create new variational state
    vstate = nk.vqs.MCState(
        sampler, model, n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain, chunk_size=chunk_size
    )
    
    # Load parameters from file
    with open(f"{filename}.mpack", 'rb') as file:
        data = file.read()
    
    vstate_loaded = flax.serialization.from_bytes(vstate, data)
    
    print(f"Model loaded from {filename}.mpack")
    
    return vstate_loaded

def log_runtime(config: Dict[str, Any], start_time: float) -> None:
    """
    Log the total runtime of the simulation.
    
    Args:
        config: Configuration dictionary
        start_time: Start time of the simulation
    """
    filename = config['filename']
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    runtime = time.time() - start_time
    data["sim_params"]["runtime"].append(str(runtime))
    
    with open(filename, 'w') as f:
        json.dump(data, f)
    
    print(f"Total runtime: {runtime:.2f} seconds")

def record_experiment_info(config: Dict[str, Any], run_id: str, description: str,
                          extra_info: Optional[Dict[str, Any]] = None) -> None:
    """
    Record experiment information to a separate file for tracking experiments.
    
    Args:
        config: Configuration dictionary
        run_id: Unique identifier for the run
        description: Description of the experiment
        extra_info: Additional information to record
    """
    # Convert config values to Python native types to ensure JSON serialization
    def convert_to_native(obj):
        """Convert JAX arrays and other non-serializable types to Python native types."""
        import numpy as np
        import jax.numpy as jnp
        
        if hasattr(obj, 'tolist'):
            # Handle NumPy and JAX arrays
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(i) for i in obj]
        elif isinstance(obj, (np.generic, jnp.ndarray)):
            return obj.item() if obj.size == 1 else obj.tolist()
        else:
            return obj

    # Convert extra_info to native types
    if extra_info:
        extra_info = convert_to_native(extra_info)
        
    # Create experiment data with native Python types
    config_native = convert_to_native(config)
    experiment_data = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": description,
        "config": {
            "Lx": config_native["Lx"],
            "bc": config_native["bc"],
            "architecture": config_native["architecture"],
            "hx": config_native["hx"],
            "hy": config_native["hy"],
            "hz": config_native["hz"],
            "J": config_native.get("J", 1.0),
            "Jy_p": config_native.get("Jy_p", 0.0),
            "Jy_v": config_native.get("Jy_v", 0.0),
            "Jbond": config_native.get("Jbond", 0.0),
            "channels_inv": config_native["channels_inv"],
            "channels_noninv": config_native["channels_noninv"],
            "n_samples": config_native["n_samples"],
            "output_files": [config_native["filename"], config_native["filename_mpack"]]
        }
    }
    
    if extra_info:
        experiment_data.update(extra_info)
    
    # Append to experiments log file
    with open("experiment_log.json", "a+") as f:
        f.seek(0)
        try:
            data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            data = []
        
        data.append(experiment_data)
        
        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=2)
    
    print(f"Experiment info recorded with run_id: {run_id}") 