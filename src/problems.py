"""The four sampling problems of the numerical experiments.

Each builder returns a SimpleNamespace with (at least)
    N, d                 number of agents and dimension
    stoch_grad(X, rng)   stochastic gradients of the local potentials f_i, X of shape (M, N, d)
    stoch_grad_central   stochastic gradient of f = sum_i f_i for the centralized PSGLD, X of shape (M, d)
    proj                 projection onto the constraint set K
    eta_max(gamma)       stepsize bound of the main theorem (same for the three networks)
plus the quantities used to evaluate each experiment.
"""
from types import SimpleNamespace
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .depsgld_core import (eta_max, proj_interval, proj_l1_ball, proj_l2_ball)
from .data import load_adult, load_housing


def _sigmoid(t):
    return 1.0 / (1.0 + np.exp(-np.clip(t, -50, 50)))


# --------------------------------------------------------------------------
# 1. Univariate constrained quartic
# --------------------------------------------------------------------------
def univariate(N=10, h=10.0, sigma=0.1, a=-1.0, b=1.0):
    """pi(x) ∝ exp(-f(x)) 1_[a,b](x), f(x) = x^2 + x^4/8 - x, split heterogeneously as
        f_i(x) = (x^2 + x^4/8)/N - b_i x,  b_i = (1 + h c_i)/N,  c_i equally spaced in [-1, 1],
    so that sum_i f_i = f. Gradients are perturbed by N(0, sigma^2) noise."""
    c = np.linspace(-1, 1, N)
    bi = (1 + h * c) / N
    f = lambda x: x**2 + x**4 / 8 - x

    def stoch_grad(X, rng):
        return (2 * X + 0.5 * X**3) / N - bi[None, :, None] + sigma * rng.standard_normal(X.shape)

    def stoch_grad_central(X, rng):
        return 2 * X + 0.5 * X**3 - 1 + sigma * np.sqrt(N) * rng.standard_normal(X.shape)

    def reference(gamma=None):
        """Quantile function, grid and density of pi (gamma=None) or of pi^gamma."""
        if gamma is None:
            g = np.linspace(a, b, 400001)
            p = np.exp(-f(g))
        else:
            g = np.linspace(a - 3, b + 3, 1600001)
            dist = np.maximum(np.maximum(a - g, g - b), 0)
            p = np.exp(-f(g) - dist**2 / (2 * gamma))
        cdf = np.cumsum(p); cdf /= cdf[-1]
        dens = p / np.trapezoid(p, g)
        return (lambda u: np.interp(u, cdf, g)), g, dens

    q_pi = reference()[0]
    u = (np.arange(4000) + 0.5) / 4000

    def w2_to_pi(samples):
        return np.sqrt(np.mean((np.quantile(samples, u) - q_pi(u)) ** 2))

    def w2_gamma_pi(gamma):
        qg = reference(gamma)[0]
        return np.sqrt(np.mean((qg(u) - q_pi(u)) ** 2))

    def mass_outside(gamma):
        _, g, dens = reference(gamma)
        return np.trapezoid(dens * ((g < a) | (g > b)), g)

    L = (2 + 1.5 * 1.5**2) / N          # local smoothness of f_i on |x| <= 1.5
    mu = 2.0 / N
    return SimpleNamespace(
        N=N, d=1, b=bi, f=f, a=a, bnd=b, stoch_grad=stoch_grad,
        stoch_grad_central=stoch_grad_central, proj=proj_interval(a, b),
        reference=reference, w2_to_pi=w2_to_pi, w2_gamma_pi=w2_gamma_pi,
        mass_outside=mass_outside, L=L, mu=mu,
        eta_max=lambda gamma: eta_max(L, mu, N, gamma))


# --------------------------------------------------------------------------
# 2. Bayesian linear regression (synthetic, l2-ball)
# --------------------------------------------------------------------------
def linear_regression(n=10_000, N=10, sig2=0.25, batch=20, s=1.4, beta_star=(1.0, 1.0), seed=11):
    """y = beta*^T x + eps, eps ~ N(0, sig2), x ~ N(0, I). Uniform prior on the l2-ball of
    radius s. The posterior is N(beta_OLS, sig2 (X^T X)^{-1}) restricted to the ball, so
    exact samples are obtained by rejection."""
    rng = np.random.default_rng(seed)
    beta_star = np.asarray(beta_star, float)
    d = len(beta_star)
    X = rng.standard_normal((n, d))
    y = X @ beta_star + np.sqrt(sig2) * rng.standard_normal(n)
    beta_ols = np.linalg.solve(X.T @ X, X.T @ y)
    perm = rng.permutation(n)
    ni = n // N
    Xa = X[perm[:N * ni]].reshape(N, ni, d)
    ya = y[perm[:N * ni]].reshape(N, ni)
    Sig = sig2 * np.linalg.inv(X.T @ X)

    def exact_samples(m, seed=0, chunk=1_000_000):
        r = np.random.default_rng(seed)
        out, cnt = [], 0
        while cnt < m:
            z = r.multivariate_normal(beta_ols, Sig, chunk)
            z = z[np.linalg.norm(z, axis=1) <= s]
            out.append(z); cnt += len(z)
        return np.concatenate(out)[:m]

    def stoch_grad(B, rng):
        M = B.shape[0]
        idx = rng.integers(0, ni, (M, N, batch))
        Xb = Xa[np.arange(N)[None, :, None], idx]
        yb = ya[np.arange(N)[None, :, None], idx]
        r = np.einsum("mnbd,mnd->mnb", Xb, B) - yb
        return (ni / batch) * np.einsum("mnbd,mnb->mnd", Xb, r) / sig2

    def stoch_grad_central(B, rng):
        M = B.shape[0]
        idx = rng.integers(0, n, (M, N * batch))
        r = np.einsum("mbd,md->mb", X[idx], B) - y[idx]
        return (n / (N * batch)) * np.einsum("mbd,mb->md", X[idx], r) / sig2

    L = max(np.linalg.eigvalsh(Xa[i].T @ Xa[i]).max() for i in range(N)) / sig2
    mu = min(np.linalg.eigvalsh(Xa[i].T @ Xa[i]).min() for i in range(N)) / sig2
    return SimpleNamespace(
        N=N, d=d, n=n, ni=ni, s=s, sig2=sig2, batch=batch, X=X, y=y, beta_star=beta_star,
        beta_ols=beta_ols, post_cov=Sig, exact_samples=exact_samples,
        stoch_grad=stoch_grad, stoch_grad_central=stoch_grad_central, proj=proj_l2_ball(s),
        L=L, mu=mu, eta_max=lambda gamma: eta_max(L, mu, N, gamma))


# --------------------------------------------------------------------------
# 3. Bayesian logistic regression (Adult, l2-ball)
# --------------------------------------------------------------------------
def adult_logistic(N=10, batch=32, tau2=100.0, shrink=0.8, seed=0):
    """Logistic likelihood with a N(0, tau2 I) prior restricted to the l2-ball of radius
    s = shrink * ||beta_sk||, where beta_sk is scikit-learn's LogisticRegression() estimate
    (default L2 penalty, C = 1). Training data split uniformly at random over N agents."""
    Xtr, Xte, ytr, yte, names = load_adult(seed=seed)
    sk = LogisticRegression(max_iter=5000).fit(Xtr[:, 1:], ytr)
    beta_sk = np.r_[sk.intercept_, sk.coef_.ravel()]
    s = shrink * np.linalg.norm(beta_sk)
    rng = np.random.default_rng(seed)
    ni = len(ytr) // N
    perm = rng.permutation(len(ytr))[:N * ni]
    Xa, ya = Xtr[perm].reshape(N, ni, -1), ytr[perm].reshape(N, ni)
    Xu, yu = Xtr[perm], ytr[perm]
    d = Xtr.shape[1]

    def stoch_grad(B, rng):
        M = B.shape[0]
        idx = rng.integers(0, ni, (M, N, batch))
        Xb = Xa[np.arange(N)[None, :, None], idx]
        yb = ya[np.arange(N)[None, :, None], idx]
        r = _sigmoid(np.einsum("mnbd,mnd->mnb", Xb, B)) - yb
        return (ni / batch) * np.einsum("mnbd,mnb->mnd", Xb, r) + B / (N * tau2)

    def stoch_grad_central(B, rng):
        M = B.shape[0]
        idx = rng.integers(0, len(yu), (M, N * batch))
        r = _sigmoid(np.einsum("mbd,md->mb", Xu[idx], B)) - yu[idx]
        return (len(yu) / (N * batch)) * np.einsum("mbd,mb->md", Xu[idx], r) + B / tau2

    def predictive(Bs):
        """Test accuracy and test NLL of the posterior predictive from draws Bs (M, d)."""
        p = _sigmoid(Xte @ Bs.T).mean(1)
        acc = np.mean((p > 0.5) == yte)
        nll = -np.mean(yte * np.log(p + 1e-12) + (1 - yte) * np.log(1 - p + 1e-12))
        return acc, nll

    p_sk = sk.predict_proba(Xte[:, 1:])[:, 1]
    sk_acc = np.mean((p_sk > 0.5) == yte)
    sk_nll = -np.mean(yte * np.log(p_sk) + (1 - yte) * np.log(1 - p_sk))
    L = max(0.25 * np.linalg.eigvalsh(Xa[i].T @ Xa[i]).max() for i in range(N)) + 1 / (N * tau2)
    mu = 1 / (N * tau2)
    return SimpleNamespace(
        N=N, d=d, ni=ni, batch=batch, tau2=tau2, s=s, beta_sk=beta_sk, names=names,
        n_train=len(ytr), n_test=len(yte), sk_acc=sk_acc, sk_nll=sk_nll,
        stoch_grad=stoch_grad, stoch_grad_central=stoch_grad_central, proj=proj_l2_ball(s),
        predictive=predictive, L=L, mu=mu, eta_max=lambda gamma: eta_max(L, mu, N, gamma))


# --------------------------------------------------------------------------
# 4. Bayesian lasso regression (California housing, l1-ball, regional agents)
# --------------------------------------------------------------------------
def housing_lasso(N=10, batch=32, tau2=100.0, shrink=0.8, test_size=0.2, seed=0):
    """Gaussian likelihood (plug-in noise variance from OLS) with a N(0, tau2 I) prior
    restricted to the l1-ball of radius s = shrink * ||beta_OLS||_1. Features standardized,
    response centered (no intercept). Agents are N regions from k-means on the coordinates;
    agent 1 is the largest region (the hub of the star)."""
    coords, Xdf, y = load_housing()
    names = list(Xdf.columns)
    X = Xdf.values
    region = KMeans(n_clusters=N, n_init=10, random_state=seed).fit_predict(coords.values)
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=test_size, random_state=seed)
    mu_x, sd_x = X[idx_tr].mean(0), X[idx_tr].std(0)
    Xs, ys = (X - mu_x) / sd_x, y - y[idx_tr].mean()
    Xtr, ytr, Xte, yte = Xs[idx_tr], ys[idx_tr], Xs[idx_te], ys[idx_te]
    order = np.argsort(-np.bincount(region[idx_tr], minlength=N))
    agent_of_region = np.empty(N, int)
    agent_of_region[order] = np.arange(N)
    agent = agent_of_region[region]              # agent index (0-based) of every district
    ag_tr = agent[idx_tr]
    n_i = np.bincount(ag_tr, minlength=N)
    Xloc = [Xtr[ag_tr == i] for i in range(N)]
    yloc = [ytr[ag_tr == i] for i in range(N)]
    d = Xtr.shape[1]
    beta_ols = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    sig2 = np.mean((ytr - Xtr @ beta_ols) ** 2) * len(ytr) / (len(ytr) - d)
    s = shrink * np.abs(beta_ols).sum()
    P = proj_l1_ball(s)

    nmax = n_i.max()
    Xpad = np.zeros((N, nmax, d)); ypad = np.zeros((N, nmax))
    for i in range(N):
        Xpad[i, :n_i[i]] = Xloc[i]; ypad[i, :n_i[i]] = yloc[i]

    def stoch_grad(B, rng):
        M = B.shape[0]
        idx = (rng.random((M, N, batch)) * n_i[None, :, None]).astype(int)
        Xb = Xpad[np.arange(N)[None, :, None], idx]
        yb = ypad[np.arange(N)[None, :, None], idx]
        r = np.einsum("mnbd,mnd->mnb", Xb, B) - yb
        return (n_i[None, :, None] / batch) * np.einsum("mnbd,mnb->mnd", Xb, r) / sig2 + B / (N * tau2)

    def stoch_grad_central(B, rng):
        M = B.shape[0]
        idx = rng.integers(0, len(ytr), (M, N * batch))
        r = np.einsum("mbd,md->mb", Xtr[idx], B) - ytr[idx]
        return (len(ytr) / (N * batch)) * np.einsum("mbd,mb->md", Xtr[idx], r) / sig2 + B / tau2

    def rmse(B):
        """Test RMSE of the prediction X_test @ B for B of shape (d,) or (M, d) (averaged)."""
        B = np.atleast_2d(B)
        return np.mean(np.sqrt(np.mean((Xte @ B.T - yte[:, None]) ** 2, axis=0)))

    def constrained_ls(Xm, yv, iters=20000):
        """argmin ||y - X b||^2 subject to ||b||_1 <= s (projected gradient), i.e. the lasso."""
        Lm = np.linalg.eigvalsh(Xm.T @ Xm).max()
        bb = np.zeros(d)
        for _ in range(iters):
            bb = P(bb - Xm.T @ (Xm @ bb - yv) / Lm)
        return bb

    L = max(np.linalg.eigvalsh(Xloc[i].T @ Xloc[i]).max() for i in range(N)) / sig2 + 1 / (N * tau2)
    mu = 1 / (N * tau2)
    return SimpleNamespace(
        N=N, d=d, n_i=n_i, batch=batch, tau2=tau2, s=s, sig2=sig2, names=names,
        coords=coords, agent=agent, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xloc=Xloc, yloc=yloc,
        beta_ols=beta_ols, stoch_grad=stoch_grad, stoch_grad_central=stoch_grad_central,
        proj=P, rmse=rmse, constrained_ls=constrained_ls,
        L=L, mu=mu, eta_max=lambda gamma: eta_max(L, mu, N, gamma))
