"""
Module for computing physical observables and quantities of interest.
"""

import json
import numpy as np
import netket as nk
import netket.experimental as nkx
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional, Union, Callable

def calculate_wilson_loops(
    vstate: nk.vqs.VariationalState,
    geometry,
    radius: int = 1,
    plot: bool = False
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Evaluate Wilson Loop and BFFM order parameters.
    
    Args:
        vstate: Variational state
        geometry: Geometry object
        radius: Radius of the Wilson loop
        plot: Whether to plot the Wilson loop
        
    Returns:
        Tuple containing:
            - WilsonX_mean: Mean of Wilson X loop
            - WilsonX_std: Standard deviation of Wilson X loop
            - X_BFFMmean: Mean of X BFFM
            - X_BFFMstd: Standard deviation of X BFFM
            - WilsonZ_mean: Mean of Wilson Z loop
            - WilsonZ_std: Standard deviation of Wilson Z loop
            - Z_BFFMmean: Mean of Z BFFM
            - Z_BFFMstd: Standard deviation of Z BFFM
    """
    center = (geometry.Lx - 1) / 2
    shift = geometry.Lx / 2 - radius - 2
    
    # Data for X loop calculations (integer radius)
    avWilsonX = []
    Xclosedlength = []
    Xopenlength = []
    X_BFFM = []
    
    # Data for Z loop calculations (non-integer radius)
    avWilsonZ = []
    Zclosedlength = []
    Zopenlength = []
    Z_BFFM = []
    
    # Calculate X Wilson loops with integer radius
    for x in np.arange(center - shift, center + shift, 1.0):
        for y in tqdm(np.arange(center - shift, center + shift, 1.0), desc="X Wilson loops"):
            # Calculate Wilson X operators
            closedstringX, openstringX, ClosedWilsonX, OpenWilsonX = wilson_loop_obs_x(
                vstate.hilbert, geometry, [x, y], radius
            )
            
            # Calculate expectation values
            ClosedWilsonX_expect = vstate.expect(ClosedWilsonX).mean
            OpenWilsonX_expect = vstate.expect(OpenWilsonX).mean
            
            # Store results
            avWilsonX.append(ClosedWilsonX_expect)
            X_BFFM.append(OpenWilsonX_expect / np.sqrt(np.abs(ClosedWilsonX_expect)))
            Xclosedlength.append(len(closedstringX))
            Xopenlength.append(len(openstringX))
    
    # Calculate Z Wilson loops with non-integer radius (radius + 0.5)
    z_radius = radius + 0.5
    for x in np.arange(center - shift, center + shift, 1.0):
        for y in tqdm(np.arange(center - shift, center + shift, 1.0), desc="Z Wilson loops"):
            try:
                # Calculate Wilson Z operators
                closedstringZ, openstringZ, ClosedWilsonZ, OpenWilsonZ = wilson_loop_obs_z(
                    vstate.hilbert, geometry, [x, y], z_radius
                )
                
                # Calculate expectation values
                ClosedWilsonZ_expect = vstate.expect(ClosedWilsonZ).mean
                OpenWilsonZ_expect = vstate.expect(OpenWilsonZ).mean
                
                # Store results
                avWilsonZ.append(ClosedWilsonZ_expect)
                Z_BFFM.append(OpenWilsonZ_expect / np.sqrt(np.abs(ClosedWilsonZ_expect)))
                Zclosedlength.append(len(closedstringZ))
                Zopenlength.append(len(openstringZ))
            except Exception as e:
                print(f"Error calculating Z Wilson loop at position [{x}, {y}], radius {z_radius}: {e}")
                continue
    
    # Compute and return X and Z statistics
    X_mean = np.mean(avWilsonX) if avWilsonX else np.nan
    X_std = np.std(avWilsonX) if avWilsonX else np.nan
    X_BFFM_mean = np.mean(X_BFFM) if X_BFFM else np.nan
    X_BFFM_std = np.std(X_BFFM) if X_BFFM else np.nan
    
    Z_mean = np.mean(avWilsonZ) if avWilsonZ else np.nan
    Z_std = np.std(avWilsonZ) if avWilsonZ else np.nan
    Z_BFFM_mean = np.mean(Z_BFFM) if Z_BFFM else np.nan
    Z_BFFM_std = np.std(Z_BFFM) if Z_BFFM else np.nan
    
    return (
        X_mean, X_std,
        X_BFFM_mean, X_BFFM_std,
        Z_mean, Z_std,
        Z_BFFM_mean, Z_BFFM_std
    )


def wilson_strings(geometry, pos: Tuple[float, float], radius: float) -> Tuple[np.ndarray, List]:
    """
    Find Wilson strings.
    
    Args:
        geometry: Geometry object
        pos: Position (x, y)
        radius: Radius around the position
        
    Returns:
        Tuple containing:
            - Closed indices: Indices for the closed Wilson loop
            - Open indices: Indices for the open Wilson string
    """
    selectedlocs_small = geometry.select_subset(pos, radius - 1/2)  # Small square ball
    selectedlocs_large = geometry.select_subset(pos, radius)  # Large square ball
    
    # Find the closed loop by subtracting small from large
    closedindices = large_subtract_small(selectedlocs_large, selectedlocs_small)
    
    # Find the open string
    openindices = half_length_wilson(selectedlocs_large, pos, radius)
    
    return closedindices, openindices


def large_subtract_small(largeball: np.ndarray, smallball: np.ndarray) -> np.ndarray:
    """
    Calculate the Wilson loop from large and small balls.
    
    Args:
        largeball: Large ball coordinates
        smallball: Small ball coordinates
        
    Returns:
        Difference between large and small balls
    """
    difference = []
    for a in largeball:
        present = False
        for b in smallball:
            if np.allclose(a, b):
                present = True
                break
        if not present:
            difference.append(a)
    
    return np.array(difference)


def half_length_wilson(largeball: np.ndarray, pos: Tuple[float, float], radius: float) -> List:
    """
    Find half-length Wilson loop string.
    
    Args:
        largeball: Large ball coordinates
        pos: Position (x, y)
        radius: Radius
        
    Returns:
        Half-length Wilson loop string
    """
    dleft = largeball[largeball[:, 0] == pos[0] - radius]
    dleft = dleft[0:int(len(dleft) / 2)]
    
    if not (len(dleft) / 2).is_integer():  # For WilsonZ case open string
        dleft = dleft[0:int(len(dleft) / 2) + 1]
    
    ddown = largeball[largeball[:, 1] == pos[1] - radius]
    dright = largeball[largeball[:, 0] == pos[0] + radius]
    dright = dright[0:int(len(dright) / 2)]
    dtop = largeball[largeball[:, 1] == pos[1] + radius]
    
    return dleft.tolist() + ddown.tolist() + dright.tolist()


def wilson_loop_obs_x(
    hi: nk.hilbert.Spin, 
    geometry, 
    pos: Tuple[float, float], 
    radius: int
) -> Tuple[np.ndarray, List, Any, Any]:
    """
    Calculate Wilson loop X expectation values.
    
    Args:
        hi: Hilbert space
        geometry: Geometry object
        pos: Position (x, y)
        radius: Radius of the Wilson loop
        
    Returns:
        Tuple containing:
            - Closed indices: Indices for the closed Wilson loop
            - Open indices: Indices for the open Wilson string
            - Closed Wilson X operator
            - Open Wilson X operator
    """
    assert float(radius).is_integer() == True, "Valid WilsonX operator only if defined on product of vertices"
    
    def X_Wilson(indices):
        """Create a product of X operators on the given indices."""
        op = 1
        for j in indices:
            op *= nk.operator.spin.sigmax(hi, j)
        return op
    
    closedindices, openindices = wilson_strings(geometry, pos, radius)
    closedstring = geometry.qubit_select(closedindices)
    openstring = geometry.qubit_select(openindices)
    
    return closedindices, openindices, X_Wilson(closedstring), X_Wilson(openstring)


def wilson_loop_obs_z(
    hi: nk.hilbert.Spin, 
    geometry, 
    pos: Tuple[float, float], 
    radius: float
) -> Tuple[np.ndarray, List, Any, Any]:
    """
    Calculate Wilson loop Z expectation values.
    
    Args:
        hi: Hilbert space
        geometry: Geometry object
        pos: Position (x, y)
        radius: Radius of the Wilson loop
        
    Returns:
        Tuple containing:
            - Closed indices: Indices for the closed Wilson loop
            - Open indices: Indices for the open Wilson string
            - Closed Wilson Z operator
            - Open Wilson Z operator
    """
    assert float(radius).is_integer() == False, "Valid WilsonZ operator only if defined on product of plaquettes"
    
    def Z_Wilson(indices):
        """Create a product of Z operators on the given indices."""
        op = 1
        for j in indices:
            op *= nk.operator.spin.sigmaz(hi, j)
        return op
    
    closedindices, openindices = wilson_strings(geometry, pos, radius)
    closedstring = geometry.qubit_select(closedindices)
    openstring = geometry.qubit_select(openindices)
    
    return closedindices, openindices, Z_Wilson(closedstring), Z_Wilson(openstring)


def calculate_renyi_entropy(
    vstate: nk.vqs.VariationalState,
    geometry,
    radius: float = 1.0
) -> Tuple[float, float, int, int]:
    """
    Calculate average Renyi entropy.
    
    Args:
        vstate: Variational state
        geometry: Geometry object
        radius: Radius of the subsystem
        
    Returns:
        Tuple containing:
            - Mean Renyi entropy
            - Standard deviation of Renyi entropy
            - Number of qubits in the subsystem
            - Perimeter of the subsystem
    """
    center = (geometry.Lx - 1) / 2
    shift = geometry.Lx / 2 - radius - 2
    renyi_mean = []
    
    if shift != 0:
        arange = np.arange(center - shift, center + shift, 1.0)
        with tqdm(total=len(arange) * len(arange), desc='ProgressBar') as pbar:
            for x in arange:
                for y in arange:
                    renyi_entropy = renyi(vstate, geometry, radius, x, y)
                    renyi_mean.append(renyi_entropy[0].mean)
                    pbar.update(1)
    else:
        renyi_entropy = renyi(vstate, geometry, radius, center, center)
        renyi_mean.append(renyi_entropy[0].mean)
    
    return np.mean(renyi_mean), np.std(renyi_mean), renyi_entropy[1], 4 * radius


def renyi(
    vstate: nk.vqs.VariationalState,
    geometry,
    radius: float,
    centerx: float = None,
    centery: float = None
) -> Tuple[Any, int]:
    """
    Calculate Renyi Entropy2 of qubits in a square centered at (centerx, centery) with the given radius.
    
    Args:
        vstate: Variational state
        geometry: Geometry object
        radius: Radius of the subsystem
        centerx: x-coordinate of the center
        centery: y-coordinate of the center
        
    Returns:
        Tuple containing:
            - Renyi entropy
            - Number of qubits in the subsystem
    """
    if centerx is None:
        centerx = (geometry.Lx - 1) / 2
    if centery is None:
        centery = (geometry.Lx - 1) / 2
        
    hi = vstate.hilbert
    
    # Select qubits in the small square ball
    selectedlocs_small = geometry.select_subset([centerx, centery], radius - 1/2)
    qubits = geometry.qubit_select(selectedlocs_small)
    
    # Calculate Renyi entropy
    renyi = nkx.observable.Renyi2EntanglementEntropy(hi, qubits)
    
    return vstate.expect(renyi), len(qubits)


def calculate_magnetizations(
    vstate: nk.vqs.VariationalState,
    geometry,
    radius: float = 1.0
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    """
    Calculate magnetizations in x, y, and z directions.
    
    Args:
        vstate: Variational state
        geometry: Geometry object
        radius: Radius of the region to consider
        
    Returns:
        Tuple containing:
            - magnetizationsXmean: Mean magnetization in x direction
            - magnetizationsXstd: Standard deviation of magnetization in x direction
            - magnetizationsYmean: Mean magnetization in y direction
            - magnetizationsYstd: Standard deviation of magnetization in y direction
            - magnetizationsZmean: Mean magnetization in z direction
            - magnetizationsZstd: Standard deviation of magnetization in z direction
    """
    center = (geometry.Lx - 1) / 2
    sel_loc = geometry.select_subset([center, center], radius)
    sel_q = geometry.qubit_select(sel_loc)
    
    magnetizationsXmean = []
    magnetizationsXstd = []
    magnetizationsYmean = []
    magnetizationsYstd = []
    magnetizationsZmean = []
    magnetizationsZstd = []
    
    hi = vstate.hilbert
    
    loop = tqdm(sel_q)
    for j in loop:
        magnetizationX = vstate.expect(nk.operator.spin.sigmax(hi, j))
        magnetizationY = vstate.expect(nk.operator.spin.sigmay(hi, j))
        magnetizationZ = vstate.expect(nk.operator.spin.sigmaz(hi, j))
        
        magnetizationsXmean.append(magnetizationX.mean)
        magnetizationsXstd.append(np.sqrt(magnetizationX.error_of_mean))
        magnetizationsYmean.append(magnetizationY.mean)
        magnetizationsYstd.append(np.sqrt(magnetizationY.error_of_mean))
        magnetizationsZmean.append(magnetizationZ.mean)
        magnetizationsZstd.append(np.sqrt(magnetizationZ.error_of_mean))
    
    return (
        magnetizationsXmean,
        magnetizationsXstd,
        magnetizationsYmean,
        magnetizationsYstd,
        magnetizationsZmean,
        magnetizationsZstd
    )


def calculate_2point_correlators(
    vstate: nk.vqs.VariationalState,
    geometry,
    radius: float = 1.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate 2-point XX and ZZ correlators.
    
    Args:
        vstate: Variational state
        geometry: Geometry object
        radius: Radius of the region to consider
        
    Returns:
        Tuple containing:
            - unique_distances: Unique distances between qubits
            - xx_average: Average XX correlator at each distance
            - zz_average: Average ZZ correlator at each distance
            - xx_std: Standard deviation of XX correlator at each distance
            - zz_std: Standard deviation of ZZ correlator at each distance
    """
    xxval = []
    zzval = []
    distval = []
    
    center = (geometry.Lx - 1) / 2
    bulklocs = geometry.select_subset([center, center], radius)
    
    hi = vstate.hilbert
    
    with tqdm(total=len(bulklocs) * (len(bulklocs) - 1), desc='ProgressBar') as pbar:
        for x in bulklocs:
            qubitx = geometry._mapping2Dto1D(geometry.arr_coord, x)[0][0]
            for y in bulklocs:
                dist = distance(x, y)
                if np.round(dist, 3) > 0.0:
                    qubity = geometry._mapping2Dto1D(geometry.arr_coord, y)[0][0]
                    xx = x_connected_2point_correlator(vstate, hi, qubitx, qubity)
                    zz = z_connected_2point_correlator(vstate, hi, qubitx, qubity)
                    xxval.append(xx)
                    zzval.append(zz)
                    distval.append(dist)
                pbar.update(1)
    
    # Calculate statistics
    unique_distances, inverse_indices = np.unique(distval, return_inverse=True)
    xxval = np.array(xxval)
    zzval = np.array(zzval)
    
    xx_average = np.array([xxval[inverse_indices == i].mean() for i in range(len(unique_distances))])
    zz_average = np.array([zzval[inverse_indices == i].mean() for i in range(len(unique_distances))])
    xx_std = np.array([xxval[inverse_indices == i].std() for i in range(len(unique_distances))])
    zz_std = np.array([zzval[inverse_indices == i].std() for i in range(len(unique_distances))])
    
    return unique_distances, xx_average, zz_average, xx_std, zz_std


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((a - b) ** 2))


def x_connected_2point_correlator(
    vstate: nk.vqs.VariationalState,
    hi: nk.hilbert.Spin,
    qubitA: int,
    qubitB: int
) -> float:
    """
    Calculate connected 2-point correlator in x direction.
    
    Args:
        vstate: Variational state
        hi: Hilbert space
        qubitA: First qubit index
        qubitB: Second qubit index
        
    Returns:
        Connected 2-point correlator
    """
    return (
        vstate.expect(nk.operator.spin.sigmax(hi, qubitA) * nk.operator.spin.sigmax(hi, qubitB)).mean - 
        vstate.expect(nk.operator.spin.sigmax(hi, qubitA)).mean * vstate.expect(nk.operator.spin.sigmax(hi, qubitB)).mean
    )


def z_connected_2point_correlator(
    vstate: nk.vqs.VariationalState,
    hi: nk.hilbert.Spin,
    qubitA: int,
    qubitB: int
) -> float:
    """
    Calculate connected 2-point correlator in z direction.
    
    Args:
        vstate: Variational state
        hi: Hilbert space
        qubitA: First qubit index
        qubitB: Second qubit index
        
    Returns:
        Connected 2-point correlator
    """
    return (
        vstate.expect(nk.operator.spin.sigmaz(hi, qubitA) * nk.operator.spin.sigmaz(hi, qubitB)).mean - 
        vstate.expect(nk.operator.spin.sigmaz(hi, qubitA)).mean * vstate.expect(nk.operator.spin.sigmaz(hi, qubitB)).mean
    )


def create_wilson_loop_callback(geometry) -> Callable:
    """
    Create a callback to calculate Wilson loops during optimization.
    
    Args:
        geometry: Geometry object
        
    Returns:
        Callback function for calculating Wilson loops
    """
    def wilson_loop_callback(vstate: nk.vqs.VariationalState, step: int, time: float, config: Dict[str, Any]) -> None:
        print(f"Step {step}: Calculating Wilson loops...")
        wilson_stats = calculate_wilson_loops(vstate, geometry, radius=1)
        
        with open(config['filename'], 'r') as f:
            data = json.load(f)
        
        # Convert to real floats
        data["order_params"]["WilsonBFFM"].append([float(x.real) for x in wilson_stats])
        
        with open(config['filename'], 'w') as f:
            json.dump(data, f)
    
    return wilson_loop_callback


def create_magnetization_callback(geometry) -> Callable:
    """
    Create a callback to calculate magnetizations during optimization.
    
    Args:
        geometry: Geometry object
        
    Returns:
        Callback function for calculating magnetizations
    """
    def magnetization_callback(vstate: nk.vqs.VariationalState, step: int, time: float, config: Dict[str, Any]) -> None:
        # Calculate every 8 steps regardless of Lx
        if step % 8 == 0:
            print(f"Step {step}: Calculating magnetizations...")
            magnetizationsXmean, magnetizationsXstd, magnetizationsYmean, magnetizationsYstd, magnetizationsZmean, magnetizationsZstd = calculate_magnetizations(vstate, geometry)
            
            with open(config['filename'], 'r') as f:
                data = json.load(f)
            
            # Convert to real floats
            data["order_params"]["magnetization_Xmean"].append([float(x.real) for x in magnetizationsXmean])
            data["order_params"]["magnetization_Xstd"].append([float(x.real) for x in magnetizationsXstd])
            data["order_params"]["magnetization_Ymean"].append([float(x.real) for x in magnetizationsYmean])
            data["order_params"]["magnetization_Ystd"].append([float(x.real) for x in magnetizationsYstd])
            data["order_params"]["magnetization_Zmean"].append([float(x.real) for x in magnetizationsZmean])
            data["order_params"]["magnetization_Zstd"].append([float(x.real) for x in magnetizationsZstd])
            
            with open(config['filename'], 'w') as f:
                json.dump(data, f)
    
    return magnetization_callback


def create_renyi_callback(geometry) -> Callable:
    """
    Create a callback to calculate Renyi entropy during optimization.
    
    Args:
        geometry: Geometry object
        
    Returns:
        Callback function for calculating Renyi entropy
    """
    def renyi_callback(vstate: nk.vqs.VariationalState, step: int, time: float, config: Dict[str, Any]) -> None:
        print(f"Step {step}: Calculating Renyi entropy...")
        renyi_mean, renyi_std, qubit_no, perimeter = calculate_renyi_entropy(vstate, geometry)
        
        with open(config['filename'], 'r') as f:
            data = json.load(f)
        
        # Convert NumPy values to Python floats and integers
        data["order_params"]["renyi2_entropy"].append([
            float(renyi_mean),
            float(renyi_std),
            int(qubit_no),
            int(perimeter)
        ])
        
        with open(config['filename'], 'w') as f:
            json.dump(data, f)
    
    return renyi_callback


def create_2point_callback(geometry) -> Callable:
    """
    Create a callback to calculate 2-point correlators during optimization.
    
    Args:
        geometry: Geometry object
        
    Returns:
        Callback function for calculating 2-point correlators
    """
    def correlator_callback(vstate: nk.vqs.VariationalState, step: int, time: float, config: Dict[str, Any]) -> None:
        print(f"Step {step}: Calculating 2-point correlators...")
        unique_distances, xx_average, zz_average, xx_std, zz_std = calculate_2point_correlators(vstate, geometry)
        
        with open(config['filename'], 'r') as f:
            data = json.load(f)
        
        # Convert NumPy arrays to lists
        data["order_params"]["2pointCorrelators"].append([
            unique_distances.tolist(),
            xx_average.tolist(),
            zz_average.tolist(),
            xx_std.tolist(),
            zz_std.tolist()
        ])
        
        with open(config['filename'], 'w') as f:
            json.dump(data, f)
    
    return correlator_callback


def create_conditional_callbacks(geometry) -> List[Callable]:
    """
    Create callbacks based on system size.
    
    Args:
        geometry: Geometry object
        
    Returns:
        List of callback functions
    """
    callbacks = []
    
    # Always add magnetization callback
    callbacks.append(create_magnetization_callback(geometry))
    
    # if geometry.Lx > 6:
    #     callbacks.append(create_wilson_loop_callback(geometry))
    #     callbacks.append(create_renyi_callback(geometry))
    #     callbacks.append(create_2point_callback(geometry))
    
    return callbacks 