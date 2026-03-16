import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Parameters
alpha = 1.0
sigma2 = 1.0

p = np.linspace(1e-4, 1 - 1e-4, 2000)

# Baseline functions
phi = (p**2 - p + 1) / 2
K_half = np.maximum(p, 1 - p) / 2  # K(1/2, p)

# Limited liability curvature (lambda=0)
g_ll = (4 * sigma2 / (alpha * (p**2 - p + 1)))**(1/3)


def compute_g(lam):
    denom = alpha * (p**2 - p + 1) - lam * np.maximum(p, 1 - p)
    return (4 * sigma2 / denom)**(1/3)


def compute_W(lam):
    """Compute W = int_0^1 K(1/2, t) g(t) dt for a given lambda."""
    g = compute_g(lam)
    integrand = K_half * g
    return np.trapezoid(integrand, p)


# Pick representative lambda values
lambdas = [0.0, 0.3, 0.6, 0.85]
Ws = [compute_W(lam) for lam in lambdas]
# W - W0 is the actual wager amount above the baseline participation value
W0 = Ws[0]

# Colors matching the existing figure style
colors = ['#666666', '#2196F3', '#FF9800', '#E53935']

# --- Figure ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

# (a) Effective cost of curvature
for i, (lam, col) in enumerate(zip(lambdas, colors)):
    phi_tilde = (alpha * (p**2 - p + 1) - lam * np.maximum(p, 1 - p)) / 2
    style = '--' if lam == 0 else '-'
    lw = 1.8 if lam == 0 else 2.2
    wager_str = rf'$W - W_0 = {Ws[i] - W0:.2f}$' if lam > 0 else r'$W = W_0$ (limited liability)'
    lab = rf'$\lambda = {lam}$, {wager_str}'
    ax1.plot(p, phi_tilde, style, color=col, linewidth=lw, label=lab)

ax1.set_xlabel(r'Base rate $p$', fontsize=12)
ax1.set_ylabel(r'$\tilde{\phi}(p)$', fontsize=13)
ax1.set_title(r'(a) Effective cost of curvature', fontsize=13)
ax1.legend(fontsize=8.5, loc='upper center')
ax1.set_xlim(0, 1)
ax1.set_ylim(bottom=0)

# (b) Optimal curvature (normalized by mean)
for i, (lam, col) in enumerate(zip(lambdas, colors)):
    g = compute_g(lam)
    g_norm = g / np.mean(g)
    style = '--' if lam == 0 else '-'
    lw = 1.8 if lam == 0 else 2.2
    wager_str = rf'$W - W_0 = {Ws[i] - W0:.2f}$' if lam > 0 else r'$W = W_0$ (limited liability)'
    lab = rf'$\lambda = {lam}$, {wager_str}'
    ax2.plot(p, g_norm, style, color=col, linewidth=lw, label=lab)

# Brier reference (constant curvature)
ax2.axhline(1.0, color='#2196F3', linewidth=1.2, linestyle=':', alpha=0.4,
            label='Constant (Brier)')

ax2.set_xlabel(r'Base rate $p$', fontsize=12)
ax2.set_ylabel(r"$G''(p)$ (normalized)", fontsize=13)
ax2.set_title(r'(b) Optimal curvature under wager', fontsize=13)
ax2.legend(fontsize=8.5, loc='upper center')
ax2.set_xlim(0, 1)

plt.tight_layout()
plt.savefig('/home/santerikoivula/Documents/thesis/main/wager_figure.pdf',
            bbox_inches='tight')
plt.savefig('/home/santerikoivula/Documents/thesis/main/wager_figure.png',
            bbox_inches='tight', dpi=150)
print("Saved wager_figure.pdf and wager_figure.png")
