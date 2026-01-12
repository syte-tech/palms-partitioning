import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph


# Import the compiled C++ module
# Ensure the .so file is in the same directory or PYTHONPATH
try:
    import palms_cpp
except ImportError:
    print("Error: Could not import 'palms_cpp'. Ensure the compilation finished successfully and the .so file is accessible.")
    sys.exit(1)

def get_dirs_and_weights(nr_dirs):
    """
    Returns directions and weights for the gradient.
    Replicates src/getDirsAndWeights.m
    """
    if nr_dirs == 2:
        dirs = np.array([[0, 1], [1, 0]])
        omegas = np.ones(2)
    elif nr_dirs == 4:
        dirs = np.array([
            [0, 1, 1, 1],
            [1, 0, 1, -1]
        ])
        omegas = np.array([np.sqrt(2)-1, np.sqrt(2)-1, 1-np.sqrt(2)/2, 1-np.sqrt(2)/2])
    else:
        raise ValueError("nr_dirs must be 2 or 4")
    return dirs, omegas

def calc_givens_angles(n, eta):
    """
    Computes recurrence coefficients (Givens rotation angles).
    Replicates src/calcGivensAngles.m
    """
    # System matrices
    A = np.zeros((2 * n, 2))
    # Fill A columns (MATLAB indices 1:2:end are 0::2 in Python)
    indices = np.arange(1, n + 1)
    A[0::2, 0] = eta * indices
    A[1::2, 0] = 1.0
    A[0::2, 1] = eta

    B = np.ones((n, 1))

    C_lin = np.zeros_like(A)
    S_lin = np.zeros_like(A)
    C_const = np.zeros_like(B)
    S_const = np.zeros_like(B)

    # Linear part
    # Note: MATLAB 1-based indexing loops i=2:size(A,1). Python 0-based is 1 to size-1.
    for i in range(1, A.shape[0]):
        limit = min(2, i) # i is 0-based index of row, so it matches MATLAB's i-1 logic effectively
        for j in range(limit):
            if A[i, j] == 0:
                continue
            
            # Givens coefficients
            rho = np.sign(A[j, j]) * np.sqrt(A[j, j]**2 + A[i, j]**2)
            c = A[j, j] / rho
            s = A[i, j] / rho
            
            C_lin[i, j] = c
            S_lin[i, j] = s
            
            # Update A
            Aj_old = A[j, :].copy()
            Ar_old = A[i, :].copy()
            A[j, :] = c * Aj_old + s * Ar_old
            A[i, :] = -s * Aj_old + c * Ar_old

    # Constant part
    for i in range(1, B.shape[0]):
        rho = np.sign(B[0, 0]) * np.sqrt(B[0, 0]**2 + B[i, 0]**2)
        c = B[0, 0] / rho
        s = B[i, 0] / rho
        
        C_const[i, 0] = c
        S_const[i, 0] = s
        
        # Update B
        B[0, :] = c * B[0, :] + s * B[i, :]

    return C_lin, S_lin, C_const, S_const

def compute_linewise_data(f, lambdas, taus, rhos, us, as_, bs, mu, nu, nr_dirs, s):
    """
    Computes data for subproblems.
    Replicates computeLinewiseData logic in src/affineLinearMS_ADMM.m
    """
    # s is 0-based index here (0 to nr_dirs-1)
    
    w_s = np.zeros_like(f)
    y_s = np.zeros_like(f)
    z_s = np.zeros_like(f)

    # Already updated splitting variables (indices 0 to s-1)
    for r in range(s):
        w_s += us[r] + lambdas[r][s] / mu
        y_s += as_[r] + taus[r][s] / nu
        z_s += bs[r] + rhos[r][s] / nu

    # Not yet updated splitting variables (indices s+1 to nr_dirs-1)
    for t in range(s + 1, nr_dirs):
        w_s += us[t] - lambdas[s][t] / mu
        y_s += as_[t] - taus[s][t] / nu
        z_s += bs[t] - rhos[s][t] / nu

    # Gather results
    denom = 2 + mu * nr_dirs * (nr_dirs - 1)
    us_data = (2 * f + mu * nr_dirs * w_s) / denom
    as_data = y_s / (nr_dirs - 1)
    bs_data = z_s / (nr_dirs - 1)

    return us_data, as_data, bs_data

def update_multipliers(us, as_, bs, lambdas, taus, rhos, mu, nu, nr_dirs):
    """Replicates updateMultipliers in src/affineLinearMS_ADMM.m"""
    for s in range(nr_dirs):
        for t in range(s + 1, nr_dirs):
            lambdas[s][t] += mu * (us[s] - us[t])
            taus[s][t] += nu * (as_[s] - as_[t])
            rhos[s][t] += nu * (bs[s] - bs[t])
    return lambdas, taus, rhos

def partitioning_stop_crit(us, as_, bs, nr_dirs, split_tol):
    """Replicates partitioningStopCrit in src/affineLinearMS_ADMM.m"""
    # Check s=0 vs s=1 (MATLAB 1 vs 2)
    def rel_diff(x, y):
        # Avoid division by zero
        denom = np.abs(x) + np.abs(y)
        mask = denom > 1e-10
        diff = np.zeros_like(x)
        diff[mask] = np.abs(x[mask] - y[mask]) / denom[mask]
        return np.max(diff)

    if rel_diff(us[0], us[1]) > split_tol: return False
    if rel_diff(as_[0], as_[1]) > split_tol: return False
    if rel_diff(bs[0], bs[1]) > split_tol: return False

    if nr_dirs > 2:
        # Check s=2 vs s=3 (MATLAB 3 vs 4)
        if rel_diff(us[2], us[3]) > split_tol: return False
        if rel_diff(as_[2], as_[3]) > split_tol: return False
        if rel_diff(bs[2], bs[3]) > split_tol: return False

    return True

def get_partitioning_from_jet_field(a, b, c, tol=1e-2):
    """
    Generates partition L from slopes a,b and offset c.
    Replicates src/getPartitioningFromJetField.m
    """
    m, n = c.shape[:2]
    num_pixels = m * n
    
    directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
    
    edges_src = []
    edges_dst = []

    # Helper to calculate relative difference
    def check_similarity(arr1, arr2):
        denom = np.abs(arr1) + np.abs(arr2)
        # Handle zeros to avoid NaN
        diff = np.zeros_like(arr1)
        mask = denom > 0
        diff[mask] = np.abs(arr1[mask] - arr2[mask]) / denom[mask]
        # Max over channels
        if diff.ndim == 3:
            return np.max(diff, axis=2)
        return diff

    # Coordinate grids
    Y, X = np.mgrid[0:m, 0:n]

    for dy, dx in directions:
        # Shift data
        # We want to compare pixel (y,x) with (y+dy, x+dx)
        # Using roll, data at (y,x) moves to (y+dy, x+dx)
        # So we compare 'data' with 'rolled_data'
        
        # Note: MATLAB circshift shifts elements. 
        # Here we manually check bounds to exclude wrapped around pixels
        
        a_shift = np.roll(a, (-dy, -dx), axis=(0, 1))
        b_shift = np.roll(b, (-dy, -dx), axis=(0, 1))
        c_shift = np.roll(c, (-dy, -dx), axis=(0, 1))
        
        # Conditions
        term1 = check_similarity(a, a_shift) > tol
        term2 = check_similarity(b, b_shift) > tol
        term3 = check_similarity(c, c_shift) > tol
        
        is_diff = term1 | term2 | term3
        
        # Valid mask (exclude boundaries where shift wraps)
        valid_y = (Y + dy >= 0) & (Y + dy < m)
        valid_x = (X + dx >= 0) & (X + dx < n)
        valid = valid_y & valid_x
        
        is_connected = (~is_diff) & valid
        
        # Get indices
        src_indices = (Y[is_connected] * n + X[is_connected])
        dst_indices = ((Y[is_connected] + dy) * n + (X[is_connected] + dx))
        
        edges_src.append(src_indices)
        edges_dst.append(dst_indices)

    # Concatenate all valid edges
    all_src = np.concatenate(edges_src)
    all_dst = np.concatenate(edges_dst)
    
    # Build sparse matrix
    data = np.ones(len(all_src), dtype=bool)
    graph = sp.coo_matrix((data, (all_src, all_dst)), shape=(num_pixels, num_pixels))
    
    # Find connected components
    n_components, labels_flat = csgraph.connected_components(graph, directed=False)
    
    return labels_flat.reshape(m, n)

def affine_linear_partitioning(f, gamma=1.0, max_iter=500, mu_nu_step=1.3, isotropic=True, split_tol=1e-2, u0=None, a0=None, b0=None, nr_threads=32, verbose=True):
    """
    Main function. Replicates affineLinearPartitioning.m
    """
    # 1. Setup
    f = f.astype(np.float64)
    m, n = f.shape[:2]
    if f.ndim == 2:
        nr_channels = 1
        f = f[:, :, np.newaxis]
    else:
        nr_channels = f.shape[2]

    nr_dirs = 4 if isotropic else 2
    
    # Defaults for initializers
    if u0 is None: u0 = f.copy()
    if a0 is None: a0 = np.zeros_like(f)
    if b0 is None: b0 = np.zeros_like(f)

    # 2. Initialization
    _, omegas = get_dirs_and_weights(nr_dirs)
    max_stripe_length = max(m, n)
    
    # Splitting variables (list of arrays)
    us = [u0.copy() for _ in range(nr_dirs)]
    as_ = [a0.copy() for _ in range(nr_dirs)]
    bs = [b0.copy() for _ in range(nr_dirs)]

    # Lagrange Multipliers (dict of arrays, keyed by (s,t))
    lambdas = [[None]*nr_dirs for _ in range(nr_dirs)]
    taus    = [[None]*nr_dirs for _ in range(nr_dirs)]
    rhos    = [[None]*nr_dirs for _ in range(nr_dirs)]
    
    for s in range(nr_dirs):
        for t in range(nr_dirs):
            lambdas[s][t] = np.zeros_like(f)
            taus[s][t]    = np.zeros_like(f)
            rhos[s][t]    = np.zeros_like(f)

    mu = 1e-3
    nu = min(450 * gamma * mu, 1.0)

    # 3. ADMM Loop
    dirs_matrix, _ = get_dirs_and_weights(nr_dirs) # Needed for solver

    for it in range(1, max_iter + 1):
        # Data weight
        eta = np.sqrt((2 + mu * nr_dirs * (nr_dirs - 1)) / (nu * nr_dirs * (nr_dirs - 1)))
        
        # Recurrence coefficients
        C_lin, S_lin, C_const, S_const = calc_givens_angles(max_stripe_length, eta)
        
        # Enforce Fortran contiguous for C++ efficiency
        C_lin = np.asfortranarray(C_lin)
        S_lin = np.asfortranarray(S_lin)
        C_const = np.asfortranarray(C_const)
        S_const = np.asfortranarray(S_const)

        for s in range(nr_dirs):
            # Jump penalty
            gamma_s = (2 * omegas[s] * gamma) / ((nr_dirs - 1) * nu)
            
            # Compute data
            us_data, as_data, bs_data = compute_linewise_data(f, lambdas, taus, rhos, us, as_, bs, mu, nu, nr_dirs, s)
            
            dir_s = dirs_matrix[:, s].astype(np.float64)

            # Call C++ Solver
            # Ensure inputs are F-contiguous
            u_out, a_out, b_out = palms_cpp.linewise_solver(
                np.asfortranarray(us_data),
                np.asfortranarray(as_data),
                np.asfortranarray(bs_data),
                np.asfortranarray(dir_s),
                gamma_s, eta,
                C_lin, S_lin, C_const, S_const,
                nr_threads
            )
            
            # Apply the swap logic here before calling C++
            # MATLAB s=1 (Python 0): x=b, y=a
            # MATLAB s=2 (Python 1): x=a, y=b
            # MATLAB s=3 (Python 2): x=a+b, y=a-b
            # MATLAB s=4 (Python 3): x=a-b, y=a+b
            
            if s == 0:
                x_in, y_in = bs_data, as_data
            elif s == 1:
                x_in, y_in = as_data, bs_data
            elif s == 2:
                x_in, y_in = as_data + bs_data, as_data - bs_data
            elif s == 3:
                x_in, y_in = as_data - bs_data, as_data + bs_data
            else:
                raise ValueError("Invalid direction index")
            
            # Call C++
            # Returns (u, x, y)
            u_res, x_res, y_res = palms_cpp.linewise_solver(
                np.asfortranarray(us_data),
                np.asfortranarray(x_in),
                np.asfortranarray(y_in),
                np.asfortranarray(dir_s),
                gamma_s, eta,
                C_lin, S_lin, C_const, S_const,
                nr_threads
            )
            
            # Back transform
            us[s] = u_res
            if s == 0:
                as_[s], bs[s] = y_res, x_res
            elif s == 1:
                as_[s], bs[s] = x_res, y_res
            elif s == 2:
                as_[s] = (x_res + y_res) / 2.0
                bs[s] = (x_res - y_res) / 2.0
            elif s == 3:
                as_[s] = (x_res + y_res) / 2.0
                bs[s] = (y_res - x_res) / 2.0

        # Update Multipliers
        lambdas, taus, rhos = update_multipliers(us, as_, bs, lambdas, taus, rhos, mu, nu, nr_dirs)
        
        # Check Stop
        if partitioning_stop_crit(us, as_, bs, nr_dirs, split_tol):
            if verbose:
                print(f"\nConverged after {it} iterations.")
            break
            
        # Update coupling
        mu *= mu_nu_step
        nu *= mu_nu_step
        if verbose:
            print(".", end="", flush=True)

    # 4. Final aggregation
    u_mean = np.mean(us, axis=0)
    a_mean = np.mean(as_, axis=0)
    b_mean = np.mean(bs, axis=0)
    
    # Calculate c (offsets)
    # c = u - x*a - y*b
    Y, X = np.mgrid[0:m, 0:n]

    # Ensure X and Y are 3D (H, W, 1) to broadcast correctly against (H, W, C)
    X = X[:, :, np.newaxis]
    Y = Y[:, :, np.newaxis]
    
    X_mat = X + 1
    Y_mat = Y + 1
    
    c_mean = u_mean - X_mat * a_mean - Y_mat * b_mean
    
    # Get Partition
    partition = get_partitioning_from_jet_field(a_mean, b_mean, c_mean)

    if u_mean.shape[2] == 1:
        u_mean = u_mean.squeeze(axis=2)
        a_mean = a_mean.squeeze(axis=2)
        b_mean = b_mean.squeeze(axis=2)
        c_mean = c_mean.squeeze(axis=2)
    
    return u_mean, partition, a_mean, b_mean, c_mean