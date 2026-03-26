import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters (same counterexample: u(d,x) = -(d-x)^2, f=1)
# Use small alpha so that the tail-optimal solution exists everywhere:
# need sqrt(2/pi) * Delta * f / (alpha * sigma * phi(p)) > 1 for all p.
alpha = 0.01
sigma2 = 1.0
sigma = np.sqrt(sigma2)
Delta = 0.1  # error threshold

p = np.linspace(1e-4, 1 - 1e-4, 2000)

# Baseline functions for the counterexample
# H(p) = -2, |H(p)| = 2, f(p) = 1, phi(p) = (p^2 - p + 1)/2
H_abs = 2.0
f = 1.0
phi = (p**2 - p + 1) / 2

# Variance-optimal curvature (eq 22)
g_var = (sigma2 * H_abs * f / (alpha * phi))**(1/3)

# Tail-optimal curvature (eq 55 in tex / eq:g-tail-optimal)
# G''_tail(p) = (sigma/Delta) * sqrt(2 * ln(sqrt(2/pi) * Delta * f / (alpha * sigma * phi)))
log_arg = np.sqrt(2 / np.pi) * Delta * f / (alpha * sigma * phi)
# Only valid where log_arg > 1; for our parameters check this holds
g_tail = (sigma / Delta) * np.sqrt(2 * np.log(log_arg))

# Brier score: constant curvature G''=2
g_brier = 2.0 * np.ones_like(p)

# --- Figure: three panels ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))

colors = {'var': '#2196F3', 'tail': '#E53935', 'brier': '#666666'}

# (a) Curvature profiles (normalized by mean for shape comparison)
g_var_norm = g_var / np.mean(g_var)
g_tail_norm = g_tail / np.mean(g_tail)

ax1.plot(p, g_var_norm, '-', color=colors['var'], linewidth=2.2,
         label='Variance-optimal')
ax1.plot(p, g_tail_norm, '-', color=colors['tail'], linewidth=2.2,
         label='Tail-optimal')
ax1.axhline(1.0, color=colors['brier'], linewidth=1.5, linestyle='--',
            label='Brier (constant)', alpha=0.6)

ax1.set_xlabel(r'Base rate $p$', fontsize=12)
ax1.set_ylabel(r"$G''(p)$ (normalized by mean)", fontsize=13)
ax1.set_title(r'(a) Curvature profiles', fontsize=13)
ax1.legend(fontsize=9)
ax1.set_xlim(0, 1)

# (b) Full objective integrand: sigma^2 |H(p)| f(p) / (2 g^2) + alpha phi(p) g
# This is what the variance-optimal rule minimizes pointwise (eq 19).
loss_var = sigma2 * H_abs * f / (2 * g_var**2) + alpha * phi * g_var
loss_tail = sigma2 * H_abs * f / (2 * g_tail**2) + alpha * phi * g_tail
loss_brier = sigma2 * H_abs * f / (2 * g_brier**2) + alpha * phi * g_brier

ax2.plot(p, loss_var, '-', color=colors['var'], linewidth=2.2,
         label='Variance-optimal')
ax2.plot(p, loss_tail, '-', color=colors['tail'], linewidth=2.2,
         label='Tail-optimal')
ax2.plot(p, loss_brier, '--', color=colors['brier'], linewidth=1.5,
         label='Brier (constant)', alpha=0.6)

ax2.set_xlabel(r'Base rate $p$', fontsize=12)
ax2.set_ylabel(r'Expected loss + cost', fontsize=13)
ax2.set_title(r'(b) Variance objective integrand', fontsize=13)
ax2.legend(fontsize=9)
ax2.set_xlim(0, 1)

# (c) Tail probability Pr(|phat - p| > Delta) under each rule
tail_var = 2 * norm.sf(Delta * g_var / sigma)   # sf = 1 - cdf = Phi_bar
tail_tail = 2 * norm.sf(Delta * g_tail / sigma)
tail_brier = 2 * norm.sf(Delta * g_brier / sigma)

ax3.plot(p, tail_var, '-', color=colors['var'], linewidth=2.2,
         label='Variance-optimal')
ax3.plot(p, tail_tail, '-', color=colors['tail'], linewidth=2.2,
         label='Tail-optimal')
ax3.plot(p, tail_brier, '--', color=colors['brier'], linewidth=1.5,
         label='Brier (constant)', alpha=0.6)

ax3.set_xlabel(r'Base rate $p$', fontsize=12)
ax3.set_ylabel(r'$\Pr(|\hat{p} - p| > \Delta)$', fontsize=13)
ax3.set_title(rf'(c) Tail probability ($\Delta = {Delta}$)', fontsize=13)
ax3.legend(fontsize=9)
ax3.set_xlim(0, 1)

plt.tight_layout()
plt.savefig('/home/santerikoivula/Documents/thesis/main/tail_figure.pdf',
            bbox_inches='tight')
plt.savefig('/home/santerikoivula/Documents/thesis/main/tail_figure.png',
            bbox_inches='tight', dpi=150)
print("Saved tail_figure.pdf and tail_figure.png")
