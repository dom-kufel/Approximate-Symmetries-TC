"""
Module defining neural network models. It has a bottom up imeplementaiton of the CNN kernels to match dual square lattice. 
"""

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from typing import List, Tuple, Dict, Any, Optional, Union, Iterable

class KernelManager:
    """
    Class to manage kernels for CNN architectures.
    
    Attributes:
        Lx (int): Lattice size in x direction
        Ly (int): Lattice size in y direction
        bc (str): Boundary conditions
        kernel_size (int): Kernel size for non-invariant CNN
        kernel_size_inv (int): Kernel size for invariant CNN
        arr_coord (np.ndarray): Array of qubit coordinates
        dg_p (Any): Dual lattice for plaquette stabilizers
        N (int): Number of qubits
    """
    
    def __init__(self, Lx: int, Ly: int, bc: str, kernel_size: int, kernel_size_inv: int, 
                 arr_coord: np.ndarray, dg_p: Any, N: int):
        """
        Initialize the kernel manager.
        
        Args:
            Lx: Lattice size in x direction
            Ly: Lattice size in y direction
            bc: Boundary conditions ('OBC' or 'PBC')
            kernel_size: Kernel size for non-invariant CNN
            kernel_size_inv: Kernel size for invariant CNN
            arr_coord: Array of qubit coordinates
            dg_p: Dual lattice for plaquette stabilizers
            N: Number of qubits
        """
        self.Lx = Lx
        self.Ly = Ly
        self.bc = bc
        self.kernel_size = kernel_size
        self.kernel_size_inv = kernel_size_inv
        self.arr_coord = arr_coord
        self.dg_p = dg_p
        self.N = N
        
        # Generate shifts
        self.shifts_links = self._generate_pos_shifts()
        self.shifts_plaq = np.array([
            [1, 1/2], [1/2, 1], [-1/2, 1], [-1, 1/2], [-1, -1/2], [-1/2, -1], [1/2, -1], [1, -1/2],
            [1/2, 0], [0, 1/2], [-1/2, 0], [0, -1/2]
        ])
        
        # Generate kernels
        self._setup_kernels()
    
    def _recursive_func(self, n: int) -> int:
        """Utility function to evaluate non-invariant kernel size."""
        if n == 0:
            return 0
        return 4 * n + self._recursive_func(n - 1)
    
    def _generate_pos_shifts(self) -> List[np.ndarray]:
        """
        Generate position shifts for CNN_links architecture.
        
        Returns:
            List of all possible ordered shifts from a link
        """
        units = self.Lx - 1
        tr_diag = np.array([1/2, 1/2]) * units
        tl_diag = np.array([-1/2, 1/2])
        up = []
        all_shifts_links = []
        
        for ring_no in range(0, units):
            left = [up[-1] + tl_diag + np.array([-1, 0]) * q if len(up) > 0
                   else tr_diag + np.array([-1, 0]) * q for q in range(0, units + 1 - ring_no)]
            down = [left[-1] + np.array([0, -1]) * q for q in range(1, units + 1 - ring_no)]
            right = [down[-1] + np.array([1, 0]) * q for q in range(1, units + 1 - ring_no)]
            up = [right[-1] + np.array([0, 1]) * q for q in range(1, units - ring_no)]
            all_shifts_links += (left + down + right + up)
            
        return all_shifts_links
    
    def _generate_kernel_shifts(self, kind: str, size: int = 1) -> np.ndarray:
        """
        Generate kernel shifts for CNN architectures.
        
        Args:
            kind: 'Combo' for CNN_links or 'RPP' for CNN_plaq
            size: Size of the kernel
            
        Returns:
            Array of kernel shifts
        """
        if kind == "Combo":
            arr = self.arr_coord
            shifts = self.shifts_links
        else:
            arr = self.dg_p.positions
            shifts = self.shifts_plaq
        
        g = self._recursive_func(size)
        shifts = shifts[(-1) * int(g):]
        
        kernel_shifts = []
        for coord in arr:
            if kind == "Combo":
                local = [self._mapping2Dto1D(coord)[0][0]]
            else:
                local = []
                
            for shift in shifts:
                shifted_coord = (coord + shift)
                if self.bc == "OBC":
                    if len(self._select_obc_subset(shifted_coord.reshape(1, 2))) != 0:
                        local.append(self._mapping2Dto1D(shifted_coord)[0][0])
                    else:
                        local.append(-1)
                else:
                    shifted_coord = shifted_coord % self.Lx
                    local.append(self._mapping2Dto1D(shifted_coord)[0][0])
                    
            kernel_shifts.append(local)
            
        return np.array(kernel_shifts)
    
    def _mapping2Dto1D(self, coord: np.ndarray) -> np.ndarray:
        """Map 2D coordinates to 1D index."""
        return np.argwhere(np.logical_and(
            self.arr_coord[:, 0] == coord[0],
            self.arr_coord[:, 1] == coord[1]
        ))
    
    def _select_obc_subset(self, arr: np.ndarray) -> np.ndarray:
        """Select OBC qubits by removing those outside the range."""
        return arr[np.logical_and(
            np.logical_and(arr[:, 0] <= self.Lx-1, arr[:, 1] <= self.Ly-1),
            np.logical_and(arr[:, 0] >= 0, arr[:, 1] >= 0)
        )]
    
    def _setup_kernels(self):
        """Setup kernels for neural network architectures."""
        # Generate kernels for architecture
        if self.bc == "OBC":
            size = self.Lx - 1
        else:
            size = self.Lx
            
        # Setup for Combo architecture
        self.kernel_shifts = self._generate_kernel_shifts(kind="Combo", size=self.kernel_size)
        
        # Find all horizontal and vertical edges
        hor_edge_lst = []
        extra = 0
        for k in range(0, self.Lx - 1):
            hor_edge_lst += list(range((self.Ly - 1) + extra, (self.Ly - 1) + self.Ly + extra))
            extra += self.Ly + (self.Ly - 1)
            
        vert_edge_lst = np.where(~np.isin(np.arange(0, len(self.kernel_shifts)), hor_edge_lst))[0].tolist()
        
        # Extract kernels for horizontal and vertical edges
        self.kernel_shifts_hor = self.kernel_shifts[hor_edge_lst]
        self.kernel_shifts_vert = self.kernel_shifts[vert_edge_lst]
        
        # Setup masks for CNN_links
        self.zero_pad_indices_hor = [jnp.argwhere(self.kernel_shifts_hor[j] == -1) 
                                     for j in range(0, len(self.kernel_shifts_hor))]
        self.zero_pad_indices_vert = [jnp.argwhere(self.kernel_shifts_vert[j] == -1) 
                                      for j in range(0, len(self.kernel_shifts_vert))]
        
        self.hor_edge_lst = jnp.array(hor_edge_lst)
        self.vert_edge_lst = jnp.array(vert_edge_lst)
        
        # Setup for RPP architecture
        self.kernel_shifts_p = self._generate_kernel_shifts(kind="plaq", size=self.kernel_size)
        self.zero_pad_indices_p = [jnp.argwhere(self.kernel_shifts_p[j] == -1) 
                                   for j in range(0, len(self.kernel_shifts_p))]
        
        # Setup for CNN_invariant
        kernel_shifts_CNN = []
        for a in range(0, size):
            for b in range(0, size):
                if self.bc == "OBC":
                    zr = [self._mapping2Dto1D_plaq((self.dg_p.positions + np.array([a, b]))[j]) 
                          for j in range(0, (size) * (size))]
                else:
                    zr = [self._mapping2Dto1D_plaq((self.dg_p.positions + np.array([a, b]))[j] % self.Lx) 
                          for j in range(0, (size) * (size))]
                
                for q in range(0, len(zr)):
                    if len(zr[q]) == 0:
                        zr[q] = -1
                    else:
                        zr[q] = zr[q][0][0]
                        
                kernel_shifts_CNN.append(zr)
                
        kernel_shifts_CNN = np.array(kernel_shifts_CNN)
        
        # Reduce kernel size for CNN_Invariant
        lst_ind = []
        for k in range(0, self.kernel_size_inv):
            lst_ind += list(range(k * (self.Ly - 1), k * (self.Ly - 1) + self.kernel_size_inv))
            
        self.kernel_shifts_CNN = kernel_shifts_CNN[:, lst_ind]
        
        # Setup masks for CNN_invariant
        self.zero_pad_indices_CNN = [jnp.argwhere(self.kernel_shifts_CNN[j] == -1) 
                                    for j in range(0, len(self.kernel_shifts_CNN))]
    
    def _mapping2Dto1D_plaq(self, coord: np.ndarray) -> np.ndarray:
        """Map 2D coordinates to 1D index for plaquettes."""
        return np.argwhere(np.logical_and(
            self.dg_p.positions[:, 0] == coord[0],
            self.dg_p.positions[:, 1] == coord[1]
        ))
    
    def mask_kernel(self, nfeat_in: int, nfeat_out: int, kernel_shifts_mod: np.ndarray, 
                   zero_pad_indices_mod: List) -> jnp.ndarray:
        """
        Create masks for kernels.
        
        Args:
            nfeat_in: Number of input features
            nfeat_out: Number of output features
            kernel_shifts_mod: Kernel shifts to mask
            zero_pad_indices_mod: Indices to zero-pad
            
        Returns:
            Masked kernel
        """
        mask_W = np.array([np.ones((nfeat_out, nfeat_in, kernel_shifts_mod.shape[1])) 
                           for j in range(0, kernel_shifts_mod.shape[0])])
        
        for j in range(0, len(zero_pad_indices_mod)):
            mask_W[j, :, :, zero_pad_indices_mod[j]] = 0
            
        return jnp.array(mask_W)


def _Wilson_4spin_plaq(x, plaq_all: List[List[int]], rescale: float, dtype: Any = jnp.float64):
    """
    Wilson loop non-linearity.
    
    Args:
        x: Input tensor
        plaq_all: List of plaquette operators
        rescale: Rescale factor
        dtype: Data type
        
    Returns:
        Wilson loop values
    """
    if dtype == "complex":
        return (rescale * jnp.prod(jnp.real(x[:, jnp.array(plaq_all)]), axis=-1) + 
                1j * rescale * jnp.prod(jnp.imag(x[:, jnp.array(plaq_all)]), axis=-1))
    else:
        return jnp.prod(x[:, jnp.array(plaq_all)], axis=-1)


def identity_initializer_CNN_links(nfeat_out: int, nfeat_in: int, kernel_shape: int, dtype: Any = complex):
    """
    Build an initializer that returns an array proportional to identity matrix.
    
    Args:
        nfeat_out: Number of output features
        nfeat_in: Number of input features
        kernel_shape: Shape of the kernel
        dtype: Data type
        
    Returns:
        Initializer function
    """
    def init(key, shape, dtype=dtype):
        mat = jnp.hstack((
            jnp.array([1], dtype=dtype),
            0 * jnp.ones((kernel_shape - 1,), dtype=dtype)
        ))
        return jnp.tile(mat, [nfeat_out, nfeat_in, 1])
    
    return init


class CNN_noninvariant(nn.Module):
    """
    Combo Architecture for non-invariant layers.
    g: E-->E via a CNN.
    
    Attributes:
        nfeatCNN_in: Number of input features
        nfeatCNN_out: Number of output features
        kernel_manager: Kernel manager
        dtype: Data type
    """
    
    nfeatCNN_in: int
    nfeatCNN_out: int
    kernel_manager: Any
    dtype: Any = jnp.float64
    
    @nn.compact
    def __call__(self, x):
        # Get kernels and masks from kernel manager
        kernel_shifts = self.kernel_manager.kernel_shifts
        kernel_shifts_hor = self.kernel_manager.kernel_shifts_hor
        kernel_shifts_vert = self.kernel_manager.kernel_shifts_vert
        zero_pad_indices_hor = self.kernel_manager.zero_pad_indices_hor
        zero_pad_indices_vert = self.kernel_manager.zero_pad_indices_vert
        hor_edge_lst = self.kernel_manager.hor_edge_lst
        vert_edge_lst = self.kernel_manager.vert_edge_lst
        
        # Create parameters
        Wconv_hor = self.param(
            'Wconv_hor',
            identity_initializer_CNN_links(self.nfeatCNN_out, self.nfeatCNN_in, kernel_shifts_hor.shape[1]),
            (self.nfeatCNN_out, self.nfeatCNN_in, kernel_shifts_hor.shape[1]),
            self.dtype
        )
        Wconv_vert = self.param(
            'Wconv_vert',
            identity_initializer_CNN_links(self.nfeatCNN_out, self.nfeatCNN_in, kernel_shifts_vert.shape[1]),
            (self.nfeatCNN_out, self.nfeatCNN_in, kernel_shifts_vert.shape[1]),
            self.dtype
        )
        bconv_hor = self.param('bconv_hor', nn.initializers.zeros_init(), (self.nfeatCNN_out,), self.dtype)
        bconv_vert = self.param('bconv_vert', nn.initializers.zeros_init(), (self.nfeatCNN_out,), self.dtype)
        
        # Create masks
        mask_Wconv_hor = self.kernel_manager.mask_kernel(
            self.nfeatCNN_in, self.nfeatCNN_out, kernel_shifts_hor, zero_pad_indices_hor
        )
        mask_Wconv_vert = self.kernel_manager.mask_kernel(
            self.nfeatCNN_in, self.nfeatCNN_out, kernel_shifts_vert, zero_pad_indices_vert
        )
        
        # Reshape input
        x = x.reshape(self.nfeatCNN_in, kernel_shifts.shape[0])
        
        # Define masked convolution operations
        def masked_conv_hor(_, pair):
            mask, ker = pair
            Wconv_masked = Wconv_hor * mask
            y = jnp.einsum("ijk,jk->i", Wconv_masked, x[:, ker])
            return None, y
        
        def masked_conv_vert(_, pair):
            mask, ker = pair
            Wconv_masked = Wconv_vert * mask
            y = jnp.einsum("ijk,jk->i", Wconv_masked, x[:, ker])
            return None, y
        
        # Apply convolutions
        _, zs_hor = jax.lax.scan(masked_conv_hor, None, (mask_Wconv_hor, kernel_shifts_hor))
        _, zs_vert = jax.lax.scan(masked_conv_vert, None, (mask_Wconv_vert, kernel_shifts_vert))
        
        # Add biases
        x_hor = zs_hor.T + jnp.tensordot(bconv_hor, jnp.ones(shape=len(mask_Wconv_hor)), axes=0)
        x_vert = zs_vert.T + jnp.tensordot(bconv_vert, jnp.ones(shape=len(mask_Wconv_vert)), axes=0)
        
        # Combine horizontal and vertical results
        x = jnp.zeros(shape=(self.nfeatCNN_out, self.kernel_manager.N))
        x = x.at[:, hor_edge_lst].set(x_hor)
        x = x.at[:, vert_edge_lst].set(x_vert)
        
        # Apply activation function
        if self.dtype == "complex":
            x = ((nn.sigmoid(jnp.real(x)) - 1/2) + 1j * (nn.sigmoid(jnp.imag(x)) - 1/2)) * (2 + 2 * jnp.e) / (jnp.e - 1)
        else:
            x = (nn.sigmoid(x) - 1/2) * (2 + 2 * jnp.e) / (jnp.e - 1)
        
        return x


class CNN_noninvariant_plaq(nn.Module):
    """
    Non-invariant layers for the RPP architecture.
    g: E-->P via a CNN.
    
    Attributes:
        nfeatCNN_in: Number of input features
        nfeatCNN_out: Number of output features
        kernel_manager: Kernel manager
        init_invariant: Initializer function
        dtype: Data type
    """
    
    nfeatCNN_in: int
    nfeatCNN_out: int
    kernel_manager: Any
    init_invariant: Any
    dtype: Any = jnp.float64
    
    @nn.compact
    def __call__(self, x):
        # Get kernels and masks from kernel manager
        kernel_shifts_p = self.kernel_manager.kernel_shifts_p
        zero_pad_indices_p = self.kernel_manager.zero_pad_indices_p
        
        # Create parameters
        Wconv_p = self.param(
            'Wconv_p',
            self.init_invariant,
            (self.nfeatCNN_out, self.nfeatCNN_in, kernel_shifts_p.shape[1]),
            self.dtype
        )
        bconv_p = self.param('bconv_p', nn.initializers.zeros_init(), (self.nfeatCNN_out,), self.dtype)
        
        # Create masks
        mask_Wconv_p = self.kernel_manager.mask_kernel(
            self.nfeatCNN_in, self.nfeatCNN_out, kernel_shifts_p, zero_pad_indices_p
        )
        
        # Reshape input
        x = x.reshape(self.nfeatCNN_in, self.kernel_manager.N)
        
        # Define masked convolution operation
        def masked_conv_p(_, pair):
            mask, ker = pair
            Wconv_masked = Wconv_p * mask
            y = jnp.einsum("ijk,jk->i", Wconv_masked, x[:, ker])
            return None, y
        
        # Apply convolution
        _, zs_p = jax.lax.scan(masked_conv_p, None, (mask_Wconv_p, kernel_shifts_p))
        
        # Add bias
        x = zs_p.T + jnp.tensordot(bconv_p, jnp.ones(shape=len(mask_Wconv_p)), axes=0)
        
        # Apply activation function
        if self.dtype == "complex":
            x = ((nn.sigmoid(jnp.real(x)) - 1/2) + 1j * (nn.sigmoid(jnp.imag(x)) - 1/2)) * (2 + 2 * jnp.e) / (jnp.e - 1)
        else:
            x = (nn.sigmoid(x) - 1/2) * (2 + 2 * jnp.e) / (jnp.e - 1)
            
        return x


class CNN_plaq_block(nn.Module):
    """
    RPP architecture block: non-invariant layer + WilsonNonlinearity.
    
    Attributes:
        conv_net_plaq: Convolutional network for plaquettes
        plaq_all: List of plaquette operators
        rescale: Rescale factor
        dtype: Data type
    """
    
    conv_net_plaq: nn.Module
    plaq_all: List[List[int]]
    rescale: float
    dtype: Any = jnp.float64
    
    def __call__(self, x):
        c1 = self.conv_net_plaq(x)
        x = jnp.expand_dims(x, axis=0)
        c2 = _Wilson_4spin_plaq(x, self.plaq_all, self.rescale, self.dtype)
        return c1 + c2


class CNN_invariant(nn.Module):
    """
    Invariant block of the network.
    g: P-->C via a CNN.
    
    Attributes:
        nfeatCNN_in: Number of input features
        nfeatCNN_out: Number of output features
        kernel_manager: Kernel manager
        bc: Boundary conditions
        dtype: Data type
    """
    
    nfeatCNN_in: int
    nfeatCNN_out: int
    kernel_manager: Any
    bc: str
    dtype: Any = jnp.float64
    
    @nn.compact
    def __call__(self, x):
        # Get kernels and masks from kernel manager
        kernel_shifts_CNN = self.kernel_manager.kernel_shifts_CNN
        zero_pad_indices_CNN = self.kernel_manager.zero_pad_indices_CNN
        
        # Create parameters
        Wconv = self.param(
            'WCNN',
            nn.initializers.normal(),
            (self.nfeatCNN_out, self.nfeatCNN_in, kernel_shifts_CNN.shape[1]),
            self.dtype
        )
        bconv = self.param('bCNN', nn.initializers.zeros_init(), (self.nfeatCNN_out,), self.dtype)
        
        # Create masks
        mask_Wconv_CNN = self.kernel_manager.mask_kernel(
            self.nfeatCNN_in, self.nfeatCNN_out, kernel_shifts_CNN, zero_pad_indices_CNN
        )
        
        # Reshape input
        x = x.reshape(self.nfeatCNN_in, kernel_shifts_CNN.shape[0])
        
        # Define masked convolution operation
        def masked_conv(_, pair):
            mask, ker = pair
            Wconv_masked = Wconv * mask
            y = jnp.einsum("ijk,jk->i", Wconv_masked, x[:, ker])
            return None, y
        
        # Apply convolution
        _, x = jax.lax.scan(masked_conv, None, (mask_Wconv_CNN, kernel_shifts_CNN))
        
        # Add bias
        if self.bc == "OBC":
            x = x.T + jnp.tensordot(bconv, jnp.ones(shape=(self.kernel_manager.Lx - 1) * 
                                                   (self.kernel_manager.Ly - 1)), axes=0)
        else:
            x = x.T + jnp.tensordot(bconv, jnp.ones(shape=(self.kernel_manager.Lx) * 
                                                  (self.kernel_manager.Ly)), axes=0)
        
        # Apply activation function
        if self.dtype == "complex":
            x = nn.elu(jnp.real(x)) + 1j * nn.elu(jnp.imag(x))
        else:
            x = nn.elu(x)
            
        return x


class Final(nn.Module):
    """Final layer that averages over all lattice points and channels."""
    
    def __call__(self, x):
        return jnp.mean(x)


class Mean_Channels(nn.Module):
    """Mean over channels for bottleneck architecture."""
    
    def __call__(self, x):
        return jnp.expand_dims(jnp.mean(x, axis=0), axis=0)


class WilsonNonlinearity(nn.Module):
    """Wilson loop non-linearity layer."""
    
    plaq_all: Tuple[Tuple[int, ...], ...]
    rescale: float
    dtype: Any = jnp.float64
    
    def __call__(self, x):
        return _Wilson_4spin_plaq(x, self.plaq_all, self.rescale, self.dtype)


class Sequential(nn.Module):
    """
    Sequential layer construction, compatible with multiple samples.
    
    Attributes:
        modules: List of modules to apply in sequence
    """
    
    modules: Iterable[callable]
    
    def NN_single_sample(self, x):
        """Process a single sample through the network."""
        for module in self.modules:
            x = module(x)
        return x
    
    def __call__(self, x):
        """Process multiple samples through the network using vmap."""
        return jax.vmap(self.NN_single_sample, in_axes=(0))(x)


def create_sequential_model(*layers):
    """Create a sequential model from a list of layers."""
    return Sequential(layers)


def create_model(config: Dict[str, Any], plaq_all: List[List[int]], kernel_manager: Any) -> nn.Module:
    """
    Create a neural network model based on configuration.
    
    Args:
        config: Configuration dictionary
        plaq_all: List of plaquette operators
        kernel_manager: Kernel manager
        
    Returns:
        Neural network model
    """
    # Parse configuration
    architecture_type = config['architecture']
    channels_noninvariant = config['channels_noninv']
    channels_invariant = config['channels_inv']
    rescale = config.get('rescale', 10**(1.5))
    dtype = config['dtype']
    init_invariant = nn.initializers.normal(stddev=1e-2)
    
    # Convert plaq_all to a tuple of tuples for hashability
    plaq_all_tuple = tuple(tuple(p) for p in plaq_all)
    
    # Create invariant layers
    inv_sequence = [
        CNN_invariant(
            repin, repout, kernel_manager, config['bc'], dtype
        ) for repin, repout in zip(channels_invariant, channels_invariant[1:])
    ]
    
    # Create full model based on architecture type
    if architecture_type == "Combo":
        # Combo architecture
        noninv_sequence = [
            CNN_noninvariant(
                repin, repout, kernel_manager, dtype
            ) for repin, repout in zip(channels_noninvariant, channels_noninvariant[1:])
        ]
        
        sequence = noninv_sequence + [
            WilsonNonlinearity(plaq_all_tuple, rescale, dtype)
        ] + inv_sequence + [Final()]
        
    else:
        # RPP architecture
        conv_net_plaq = CNN_noninvariant_plaq(
            *channels_noninvariant, kernel_manager, init_invariant, dtype
        )
        
        sequence = [
            CNN_plaq_block(conv_net_plaq, plaq_all_tuple, rescale, dtype)
        ] + inv_sequence + [Final()]
    
    return create_sequential_model(*sequence) 