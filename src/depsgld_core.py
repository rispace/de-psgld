"""Core tools for DE-PSGLD: networks, samplers, projections, and Wasserstein-2 distances.

All samplers are vectorized over M independent replicas, so that the distribution of
every agent at every iteration can be estimated from the replicas.
"""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

NETWORK_NAMES = {"full": "Fully Connected", "ring": "Circular", "star": "Star"}


# --------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------
def adjacency(kind, N):
    """Adjacency matrix of the fully connected ('full'), circular ('ring') or star ('star')
    graph on N nodes. Node 0 is the hub of the star."""
    A = np.zeros((N, N))
    if kind == "full":
        A = np.ones((N, N)) - np.eye(N)
    elif kind == "ring":
        for i in range(N):
            A[i, (i + 1) % N] = A[(i + 1) % N, i] = 1
    elif kind == "star":
        A[0, 1:] = A[1:, 0] = 1
    else:
        raise ValueError(f"unknown network '{kind}'")
    return A


def mixing_matrix(kind, N):
    """W = I - delta L with delta = 1/(d_max + 1).

    Returns (W, rho, lambda_N(W)), where rho = max(|lambda_2|, |lambda_N|).
    Since lambda_N(L) <= 2 d_max, we have 0 < delta < 2/lambda_N(L), so W is symmetric,
    doubly stochastic, with positive diagonal.
    """
    A = adjacency(kind, N)
    deg = A.sum(1)
    L = np.diag(deg) - A
    W = np.eye(N) - L / (deg.max() + 1.0)
    ev = np.sort(np.linalg.eigvalsh(W))[::-1]
    rho = max(abs(ev[1]), abs(ev[-1])) if N > 1 else 0.0
    return W, rho, ev[-1]


def eta_max(L, mu, N, gamma, kinds=tuple(NETWORK_NAMES)):
    """Stepsize bound of the main theorem,
    eta_max = min{ 2N/L_gamma, (1 + lambda_N(W))/L_gamma, 1/(L_gamma + mu) },
    with L_gamma = L + 2/(N gamma). The minimum is taken over the networks in `kinds`
    so that all networks use the same stepsize."""
    Lg = L + 2.0 / (N * gamma)
    lmin = min(mixing_matrix(k, N)[2] for k in kinds)
    return min(2 * N / Lg, (1 + lmin) / Lg, 1 / (Lg + mu))


# --------------------------------------------------------------------------
# Samplers
# --------------------------------------------------------------------------
def _progress(k, n_iter, t0, label):
    if k == n_iter or k % max(1, n_iter // 20) == 0:
        print(f"\r  {label}: {k}/{n_iter} iterations ({time.time() - t0:.0f}s)",
              end="\n" if k == n_iter else "", flush=True)


def de_psgld(x0, W, stoch_grad, proj, eta, gamma, n_iter, seed=0,
             record=None, record_every=1, progress=None):
    """Decentralized proximal SGLD:

        x_i <- sum_j W_ij x_j - eta [ g_i(x_i) + (x_i - P_K(x_i)) / (N gamma) ] + sqrt(2 eta) w_i

    x0           : array (M, N, d), initial states of M replicas of the N agents
    W            : (N, N) mixing matrix
    stoch_grad   : function (X, rng) -> (M, N, d), unbiased estimates of grad f_i(x_i)
    proj         : function X -> projection of X onto K along the last axis
    gamma        : proximal parameter (np.inf gives DE-SGLD without constraint)
    record       : optional function (X, k) -> value, called every `record_every` iterations
    progress     : optional label; if given, prints the progress
    Returns (final states, list of recorded values).
    """
    rng = np.random.default_rng(seed)
    X = np.array(x0, dtype=float, copy=True)
    M, N, d = X.shape
    out = [record(X, 0)] if record is not None else []
    t0 = time.time()
    for k in range(1, n_iter + 1):
        g = stoch_grad(X, rng)
        prox = (X - proj(X)) / (N * gamma)
        X = (np.einsum("ij,mjd->mid", W, X) - eta * (g + prox)
             + np.sqrt(2 * eta) * rng.standard_normal(X.shape))
        if record is not None and k % record_every == 0:
            out.append(record(X, k))
        if progress:
            _progress(k, n_iter, t0, progress)
    return X, out


def psgld(x0, stoch_grad, proj, eta, gamma, n_iter, seed=0,
          record=None, record_every=1, progress=None):
    """Centralized proximal SGLD on u^gamma = f + q_K^gamma:

        x <- x - eta [ g(x) + (x - P_K(x)) / gamma ] + sqrt(2 eta) w

    x0 : array (M, d). With eta = eta_DE / N this is the centralized proximal Langevin
    algorithm that the mean of DE-PSGLD tracks in the proof of the main theorem.
    """
    rng = np.random.default_rng(seed)
    X = np.array(x0, dtype=float, copy=True)
    out = [record(X, 0)] if record is not None else []
    t0 = time.time()
    for k in range(1, n_iter + 1):
        g = stoch_grad(X, rng)
        X = (X - eta * (g + (X - proj(X)) / gamma)
             + np.sqrt(2 * eta) * rng.standard_normal(X.shape))
        if record is not None and k % record_every == 0:
            out.append(record(X, k))
        if progress:
            _progress(k, n_iter, t0, progress)
    return X, out


# --------------------------------------------------------------------------
# Projections onto K
# --------------------------------------------------------------------------
def proj_interval(a, b):
    """Projection onto [a, b]."""
    return lambda X: np.clip(X, a, b)


def proj_l2_ball(radius):
    """Projection onto {x : ||x||_2 <= radius} along the last axis."""
    def P(X):
        nrm = np.linalg.norm(X, axis=-1, keepdims=True)
        return X * np.minimum(1.0, radius / np.maximum(nrm, 1e-300))
    return P


def proj_l1_ball(radius):
    """Projection onto {x : ||x||_1 <= radius} along the last axis
    (sorting algorithm of Duchi, Shalev-Shwartz, Singer and Chandra, 2008)."""
    def P(X):
        shp = X.shape
        V = X.reshape(-1, shp[-1])
        a = np.abs(V)
        inside = a.sum(1) <= radius
        u = -np.sort(-a, axis=1)
        css = np.cumsum(u, axis=1) - radius
        k = np.arange(1, V.shape[1] + 1)
        cond = u - css / k > 0
        r = cond.shape[1] - 1 - np.argmax(cond[:, ::-1], axis=1)
        theta = css[np.arange(len(V)), r] / (r + 1)
        Y = np.sign(V) * np.maximum(a - theta[:, None], 0)
        Y[inside] = V[inside]
        return Y.reshape(shp)
    return P


# --------------------------------------------------------------------------
# Wasserstein-2 distance
# --------------------------------------------------------------------------
def w2_1d(samples, ref_quantile, n_q=4000):
    """W2 between the empirical law of 1D samples and a reference law given by its
    quantile function: W2^2 = int_0^1 (F^{-1}(u) - G^{-1}(u))^2 du on n_q levels."""
    u = (np.arange(n_q) + 0.5) / n_q
    return np.sqrt(np.mean((np.quantile(samples, u) - ref_quantile(u)) ** 2))


def w2_empirical(A, B):
    """Exact W2 between two empirical measures with the same number of points
    (optimal assignment)."""
    C = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(C)
    return np.sqrt(C[r, c].mean())
