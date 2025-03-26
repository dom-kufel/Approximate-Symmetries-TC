"""
Module for time-dependent variational principle (TDVP) optimization.
"""

import jax
import jax.numpy as jnp
import netket as nk
from functools import partial
from tqdm import tqdm
from typing import Dict, Any, Callable, Optional, List, Tuple

from utils.config import update_data

def run_tdvp(
    hamiltonian: nk.operator.AbstractOperator,
    vstate: nk.vqs.VariationalState,
    config: Dict[str, Any],
    callbacks: Optional[List[Callable]] = None
) -> nk.vqs.VariationalState:
    """
    Run the time-dependent variational principle optimization.
    
    Args:
        hamiltonian: Hamiltonian operator
        vstate: Variational state
        config: Configuration dictionary
        callbacks: List of callback functions to call after each optimization step
        
    Returns:
        Optimized variational state
    """
    dt = config['dt']
    t_start = 0.0
    t_end = config['sim_time']
    diag_shift = config['diag_shift']
    filename = config['filename']
    
    n_iter = int((t_end - t_start) / dt)
    diag_scale = 0.0
    rtol = 1e-30
    rtol_smooth = 1e-30
    
    loop = tqdm(range(n_iter))
    t = t_start
    
    for step in loop:
        # Compute energy and gradient
        E, f = vstate.expect_and_grad(hamiltonian)
        
        # Compute quantum geometric tensor (QGT)
        S = vstate.quantum_geometric_tensor(
            nk.optimizer.qgt.QGTJacobianDense(diag_shift=diag_shift, diag_scale=diag_scale)
        )
        
        # Compute update direction
        gamma_f = jax.tree.map(lambda x: -1.0 * x, f)
        dtheta, _ = S.solve(
            partial(nk.optimizer.solver.pinv_smooth, rtol=rtol, rtol_smooth=rtol_smooth),
            gamma_f
        )
        
        # Update parameters
        vstate.parameters = jax.tree.map(lambda x, y: x + dt * y, vstate.parameters, dtheta)
        
        # Save optimization data
        update_data(filename, [
            "iters", "energy", "energy_eom", "energy_var", "tau_corr", 
            "Rsplit", "Vscore", "MCMC_accepted", "MCMC_total"
        ], [
            t, E.mean, E.error_of_mean, E.variance, E.tau_corr,
            E.R_hat, config['N'] * E.variance / E.mean**2,
            vstate.sampler_state.n_accepted, vstate.sampler_state.n_steps
        ])
        
        # Check for NaN values
        if jnp.isnan(E.mean):
            print("Encountered NaN energy, stopping optimization.")
            break
        
        # Call any callback functions
        if callbacks is not None and step % 8 == 0:
            for callback in callbacks:
                callback(vstate, step, t, config)
        
        # Update progress bar description
        loop.set_description(f"Energy: {E.mean:.6f} ± {E.error_of_mean:.6f}")
        
        # Update time
        t = t + dt
    
    return vstate


def create_final_callback(calculation_callbacks: List[Callable]) -> Callable:
    """
    Create a final callback function that runs all provided calculation callbacks.
    
    Args:
        calculation_callbacks: List of calculation callbacks to run
        
    Returns:
        Callback function that runs all provided callbacks
    """
    def final_callback(vstate: nk.vqs.VariationalState, config: Dict[str, Any]) -> None:
        print("Running final calculations...")
        for callback in calculation_callbacks:
            callback(vstate, -1, -1, config)
    
    return final_callback 