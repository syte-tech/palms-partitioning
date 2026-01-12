import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.spatial

from typing import Tuple

# Import the compiled C++ module
# Ensure the .so file is in the same directory or PYTHONPATH
try:
    import palms_cpp
except ImportError:
    print("Error: Could not import 'palms_cpp'. Ensure the compilation finished successfully and the .so file is accessible.")
    sys.exit(1)


def estimate_grid_orientation(points: np.ndarray, sample_size: int = 5000) -> float:
    """
    Estimates the principal rotation angle of the point lattice (grid).
    Returns angle in radians [0, pi/2).
    """
    if len(points) < 10: return 0.0
    
    # Sample points
    idx = np.random.choice(len(points), min(len(points), sample_size), replace=False)
    sample = points[idx, :2] # XY only
    
    # Find nearest neighbor for each point (k=2 because k=1 is self)
    tree = scipy.spatial.KDTree(sample)
    # Get vector to neighbor
    # tree.query gives distances/indices. We need actual vectors.
    # So we query, get index of neighbor, and subtract.
    _, neighbor_idx = tree.query(sample, k=2)
    
    # Vectors from point -> neighbor
    # neighbor_idx[:, 1] is the index of the closest neighbor
    vectors = sample[neighbor_idx[:, 1]] - sample
    
    # Calculate angles
    # arctan2 returns [-pi, pi]
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    
    # Map all angles to [0, 90) degrees (modulo pi/2)
    # We assume the grid is orthogonal, so 0, 90, 180, 270 are the "same" orientation
    angles_mod = np.mod(angles, np.pi / 2)
    
    # Histogram to find peak
    # We use a histogram with wrap-around logic or just simple bins
    hist, bin_edges = np.histogram(angles_mod, bins=90, range=(0, np.pi/2))
    
    # Smooth histogram to reduce noise
    hist_smooth = np.convolve(hist, np.ones(5)/5, mode='same')
    
    peak_idx = np.argmax(hist_smooth)
    peak_angle = (bin_edges[peak_idx] + bin_edges[peak_idx+1]) / 2
    
    # print(f"Detected Grid Orientation: {np.degrees(peak_angle):.1f} degrees")
    return peak_angle


def create_stripes(points: np.ndarray, dir_vec: Tuple[float, float], resolution: float):
    """
    Segments points into linear stripes based on a direction vector.
    
    Args:
        points: (N, 2) or (N, 3) array of points.
        dir_vec: (u, v) normalized direction vector of the stripe orientation.
        resolution: The grid resolution. Used to calculate dynamic gap threshold.
    """
    bin_size = resolution * 1.1

    u, v = dir_vec
    sec = points[:, 0]*u + points[:, 1]*v
    main = points[:, 0]*(-v) + points[:, 1]*u
    
    bins = np.floor((main - main.min()) / bin_size).astype(int)
    order = np.lexsort((sec, bins))
    sorted_idx = np.arange(len(points))[order]
    
    # On a grid, the geometric distance between connected neighbors along vector (u, v)
    # is exactly resolution / max(|u|, |v|).
    # We add a tolerance to bridge single missing pixels (holes) but break on larger gaps.
    expected_step = resolution / max(abs(u), abs(v)) if max(abs(u), abs(v)) > 0 else resolution
    gap_threshold = expected_step * 1.5
    
    bin_chg = np.diff(bins[order]) != 0
    dist_chg = np.diff(sec[order]) > gap_threshold
    breaks = np.where(bin_chg | dist_chg)[0] + 1

    stripes = np.split(sorted_idx, breaks)
    
    return [s for s in stripes]


def calc_givens_angles(n, eta):
    A = np.zeros((2 * n, 2))
    indices = np.arange(1, n + 1)
    A[0::2, 0] = eta * indices; A[1::2, 0] = 1.0; A[0::2, 1] = eta
    B = np.ones((n, 1))
    C_lin = np.zeros_like(A); S_lin = np.zeros_like(A)
    C_const = np.zeros_like(B); S_const = np.zeros_like(B)

    for i in range(1, A.shape[0]):
        limit = min(2, i)
        for j in range(limit):
            if A[i, j] == 0: continue
            rho = np.sign(A[j, j]) * np.sqrt(A[j, j]**2 + A[i, j]**2)
            c = A[j, j] / rho; s = A[i, j] / rho
            C_lin[i, j] = c; S_lin[i, j] = s
            Aj_old = A[j, :].copy(); Ar_old = A[i, :].copy()
            A[j, :] = c * Aj_old + s * Ar_old; A[i, :] = -s * Aj_old + c * Ar_old

    for i in range(1, B.shape[0]):
        rho = np.sign(B[0, 0]) * np.sqrt(B[0, 0]**2 + B[i, 0]**2)
        c = B[0, 0] / rho; s = B[i, 0] / rho
        C_const[i, 0] = c; S_const[i, 0] = s
        B[0, :] = c * B[0, :] + s * B[i, :]
    return C_lin, S_lin, C_const, S_const


def compute_linewise_data_sparse(f, lambdas, taus, rhos, us, as_, bs, mu, nu, nr_dirs, s):
    w_s = np.zeros_like(f); y_s = np.zeros_like(f); z_s = np.zeros_like(f)
    for r in range(s):
        w_s += us[r] + lambdas[r][s] / mu
        y_s += as_[r] + taus[r][s] / nu
        z_s += bs[r] + rhos[r][s] / nu
    for t in range(s + 1, nr_dirs):
        w_s += us[t] - lambdas[s][t] / mu
        y_s += as_[t] - taus[s][t] / nu
        z_s += bs[t] - rhos[s][t] / nu
    denom = 2 + mu * nr_dirs * (nr_dirs - 1)
    return (2 * f + mu * nr_dirs * w_s) / denom, y_s / (nr_dirs - 1), z_s / (nr_dirs - 1)


def get_partitioning_sparse(points_xyz, a_global, b_global, c_global, tol=1e-2):
    """
    Standard partitioning based on k-Nearest Neighbors.
    Edges are kept only if Jet values (a,b,c) are similar.
    """
    N = len(points_xyz)
    # k=8 matches the 8-neighborhood of the image version
    tree = scipy.spatial.KDTree(points_xyz[:, :2])
    dists, indices = tree.query(points_xyz[:, :2], k=9) # k=9 because self is included
    
    # Create edges (Source -> Target)
    # We flatten the neighbor array
    u_idx = np.repeat(np.arange(N), 8) # Source
    v_idx = indices[:, 1:].flatten()   # Target (skip self at col 0)
    
    # Filter invalid (if k < 9 found)
    mask = v_idx < N
    u_idx = u_idx[mask]
    v_idx = v_idx[mask]

    def check_diff(arr):
        val_u = arr[u_idx, 0]
        val_v = arr[v_idx, 0]
        denom = np.abs(val_u) + np.abs(val_v)
        mask_denom = denom > 1e-9
        diff = np.zeros_like(val_u)
        diff[mask_denom] = np.abs(val_u[mask_denom] - val_v[mask_denom]) / denom[mask_denom]
        return diff

    diff_a = check_diff(a_global)
    diff_b = check_diff(b_global)
    diff_c = check_diff(c_global)
    
    is_connected = (diff_a <= tol) & (diff_b <= tol) & (diff_c <= tol)
    
    valid_u = u_idx[is_connected]
    valid_v = v_idx[is_connected]
    
    data = np.ones(len(valid_u), dtype=bool)
    graph = sp.coo_matrix((data, (valid_u, valid_v)), shape=(N, N))
    n_components, labels = csgraph.connected_components(graph, directed=False)
    
    return labels


def palms_point_cloud(points, resolution, gamma=1.0, max_iter=100, nr_directions=8, verbose=True):
    """
    PALMS for Point Clouds with arbitrary number of directions.
    Increasing nr_directions improves robustness against grid artifacts.
    """
    N = len(points)
    nr_channels = 1
    f = points[:, 2:3].astype(np.float64)
    
    # 1. Generate Directions & Stripes
    # Angles from 0 to 180 degrees (0 to pi)
    angles = np.linspace(0, np.pi, nr_directions, endpoint=False)
    # Direction vectors: [cos(theta), sin(theta)]
    dirs = [np.array([np.cos(theta), np.sin(theta)]) for theta in angles]
    
    all_stripes = []
    if verbose: print(f"Generating stripes for {nr_directions} directions...")
    
    for d in dirs:
        s = create_stripes(points, d, resolution=resolution)
        all_stripes.append(s)
        
    nr_dirs = len(all_stripes)
    max_len = max([len(s) for dirs in all_stripes for s in dirs] + [100])

    # 2. ADMM Init
    u0 = f.copy(); a0 = np.zeros_like(f); b0 = np.zeros_like(f)
    us = [u0.copy() for _ in range(nr_dirs)]
    as_ = [a0.copy() for _ in range(nr_dirs)]
    bs = [b0.copy() for _ in range(nr_dirs)]
    lambdas = [[np.zeros_like(f) for _ in range(nr_dirs)] for _ in range(nr_dirs)]
    taus    = [[np.zeros_like(f) for _ in range(nr_dirs)] for _ in range(nr_dirs)]
    rhos    = [[np.zeros_like(f) for _ in range(nr_dirs)] for _ in range(nr_dirs)]
    
    mu = 1e-3
    nu = min(450 * gamma * mu, 1.0)
    
    # 3. Optimization Loop
    for it in range(1, max_iter + 1):
        eta = np.sqrt((2 + mu * nr_dirs * (nr_dirs - 1)) / (nu * nr_dirs * (nr_dirs - 1)))
        
        C_lin, S_lin, C_const, S_const = calc_givens_angles(max_len + 10, eta)
        C_lin = np.asfortranarray(C_lin); S_lin = np.asfortranarray(S_lin)
        C_const = np.asfortranarray(C_const); S_const = np.asfortranarray(S_const)
        
        for s in range(nr_dirs):
            gamma_s = (2 * 1.0 * gamma) / ((nr_dirs - 1) * nu)
            
            us_data, as_data, bs_data = compute_linewise_data_sparse(f, lambdas, taus, rhos, us, as_, bs, mu, nu, nr_dirs, s)
            
            # Map Slopes: Global (a,b) -> Local (Parallel, Perp)
            # Parallel (p) = a*cos + b*sin
            # Perp (q)     = -a*sin + b*cos
            cos_t = dirs[s][0]
            sin_t = dirs[s][1]
            
            in_par  =  as_data * cos_t + bs_data * sin_t
            in_perp = -as_data * sin_t + bs_data * cos_t
            
            list_u, list_a, list_b = [], [], []
            current_stripes = all_stripes[s]
            
            for idx in current_stripes:
                d_u = us_data[idx].T.copy(order='F')
                d_par = in_par[idx].T.copy(order='F')
                d_perp = in_perp[idx].T.copy(order='F')
                
                list_u.append((idx.astype(np.uint64), d_u))
                list_a.append((idx.astype(np.uint64), d_par))
                list_b.append((idx.astype(np.uint64), d_perp))
            
            dummy = np.zeros((N,1,1), order='F')
            u_res, p_res, q_res = palms_cpp.sparse_solver(
                N, nr_channels, dummy, dummy, dummy, 
                list_u, list_a, list_b, gamma_s, eta, C_lin, S_lin, C_const, S_const
            )
            
            u_out = np.array(u_res).reshape(N,1)
            p_out = np.array(p_res).reshape(N,1)
            q_out = np.array(q_res).reshape(N,1)
            
            us[s] = u_out
            
            # Rotate Back: Local -> Global
            # a = p*cos - q*sin
            # b = p*sin + q*cos
            as_[s] = p_out * cos_t - q_out * sin_t
            bs[s] = p_out * sin_t + q_out * cos_t

        # Multipliers
        for s in range(nr_dirs):
            for t in range(s + 1, nr_dirs):
                lambdas[s][t] += mu * (us[s] - us[t])
                taus[s][t]    += nu * (as_[s] - as_[t])
                rhos[s][t]    += nu * (bs[s] - bs[t])
        
        if verbose: print(".", end="", flush=True)
        mu *= 1.3; nu *= 1.3

    # 4. Final Aggregation
    u_mean = np.mean(us, axis=0)
    a_mean = np.mean(as_, axis=0)
    b_mean = np.mean(bs, axis=0)
    c_mean = u_mean - a_mean * points[:, 0:1] - b_mean * points[:, 1:2]
    
    if verbose: print("\nGenerating Partition...")
    partition = get_partitioning_sparse(points, a_mean, b_mean, c_mean)
    
    return u_mean, partition, a_mean, b_mean, c_mean