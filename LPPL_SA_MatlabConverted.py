from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import math
import random
from copy import deepcopy

# =========================
# 1. LOAD EXCEL FILE
# =========================
print("Upload your Excel file with columns 'Date' and 'Index1'")
uploaded = files.upload()
filename = list(uploaded.keys())[0]

df = pd.read_excel(filename, engine='openpyxl')
if 'Date' not in df.columns or 'Index1' not in df.columns:
    raise ValueError("Excel must contain 'Date' and 'Index1' columns")

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()
df['Index1'] = df['Index1'].fillna(method='ffill').fillna(method='bfill')
df = df.dropna()

dates = df.index
prices = df['Index1'].values
t0 = dates[0]
t = np.array([(d - t0).days for d in dates])          # days since start
y = np.log(prices)                                     # log price

# =========================
# 2. JLS / LPPL MODEL (FILIMONOV-SORNETTE LINEARIZATION)
# =========================
def build_design_matrix(t, tc, m, omega):
    dt = tc - t
    dt = np.maximum(dt, 1e-8)
    f = dt ** m
    cos_term = np.cos(omega * np.log(dt))
    sin_term = np.sin(omega * np.log(dt))
    X = np.column_stack([np.ones_like(t), f, f * cos_term, f * sin_term])
    return X

def linear_parameters(t, y, tc, m, omega):
    X = build_design_matrix(t, tc, m, omega)
    try:
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    except:
        coeffs = [np.nan]*4
    return coeffs

def rss_from_nonlinear(t, y, tc, m, omega):
    coeffs = linear_parameters(t, y, tc, m, omega)
    if np.any(np.isnan(coeffs)):
        return 1e20
    A, B, C1, C2 = coeffs
    penalty = 0.0
    if B >= 0:
        penalty += 1e6 * (B + 1.0)**2
    ampl = np.hypot(C1, C2)
    if ampl >= 1.0:
        penalty += 1e6 * (ampl - 1.0)**2
    X = build_design_matrix(t, tc, m, omega)
    y_pred = X @ coeffs
    rss = np.sum((y - y_pred)**2)
    return rss + penalty

# =========================
# 3. PARAMETER BOUNDS (non-linear only: tc, m, omega)
# =========================
t_last = t[-1]
lb_nonlin = [t_last + 1.0, 0.01, 0.5]
ub_nonlin = [t_last + 500, 0.99, 15.0]

def random_initial_nonlin():
    tc = random.uniform(t_last + 5, t_last + 200)
    m = random.uniform(0.2, 0.8)
    omega = random.uniform(1.0, 10.0)
    return [tc, m, omega]

# =========================
# 4. PERTURBATION
# =========================
def perturb_nonlin(params, T):
    new = deepcopy(params)
    scales = [20.0 * T, 0.1 * T, 1.0 * T]   # scales for tc, m, omega
    for i in range(3):
        new[i] += np.random.normal(0, scales[i])
        new[i] = max(lb_nonlin[i], min(ub_nonlin[i], new[i]))
    if new[0] <= t_last:
        new[0] = t_last + 0.1
    return new

# =========================
# 5. LOCAL REFINEMENT (Levenberg-Marquardt)
# =========================
def residuals_for_lm(params, t, y):
    tc, m, omega = params
    coeffs = linear_parameters(t, y, tc, m, omega)
    if np.any(np.isnan(coeffs)):
        return np.full_like(y, 1e10)
    X = build_design_matrix(t, tc, m, omega)
    y_pred = X @ coeffs
    return y - y_pred

def local_refine(params, t, y):
    try:
        res = least_squares(residuals_for_lm, params, bounds=(lb_nonlin, ub_nonlin),
                            args=(t, y), max_nfev=50, method='lm')
        return res.x.tolist()
    except:
        return params

# =========================
# 6. ORDINARY SIMULATED ANNEALING (Metropolis)
# =========================
def solve_lppl_sa(t, y, iterations=4000, restart_after=800,
                  T0=10.0, alpha=0.995):
    # Initial random non-linear parameters
    best = random_initial_nonlin()
    best_rss = rss_from_nonlinear(t, y, *best)
    print(f"Initial RSS: {best_rss:.4f}")

    current = deepcopy(best)
    current_rss = best_rss

    T = T0
    history = [best_rss]
    no_improve = 0

    for it in range(iterations):
        # Perturb parameters
        candidate = perturb_nonlin(current, T)
        candidate_rss = rss_from_nonlinear(t, y, *candidate)
        delta = candidate_rss - current_rss

        # Metropolis acceptance
        accept = False
        if delta < 0:
            accept = True
        else:
            if random.random() < math.exp(-delta / T):
                accept = True

        if accept:
            current = deepcopy(candidate)
            current_rss = candidate_rss
            if candidate_rss < best_rss:
                best = deepcopy(candidate)
                best_rss = candidate_rss
                no_improve = 0
                print(f"Iter {it:5d} | T={T:.3f} | RSS={best_rss:.4f}")
                # Local refinement after improvement
                best = local_refine(best, t, y)
                best_rss = rss_from_nonlinear(t, y, *best)
            else:
                no_improve += 1
        else:
            no_improve += 1

        # Cooling
        T *= alpha
        if T < 1e-6:
            T = 1e-6

        history.append(best_rss)

        # Restart if stagnation
        if no_improve > restart_after:
            print(f"Restart at iteration {it} (no improvement for {restart_after})")
            current = random_initial_nonlin()
            current_rss = rss_from_nonlinear(t, y, *current)
            no_improve = 0
            T = T0

    return best, best_rss, history

# =========================
# 7. RUN ESTIMATION
# =========================
best_nonlin, best_rss, hist = solve_lppl_sa(t, y, iterations=4000, restart_after=800, T0=10.0, alpha=0.995)

# Retrieve final linear coefficients
A, B, C1, C2 = linear_parameters(t, y, *best_nonlin)
tc, m, omega = best_nonlin
amplitude = np.hypot(C1, C2)
phi = np.arctan2(C2, C1)

print("\n" + "="*60)
print("JLS / LPPL ESTIMATION RESULTS (Ordinary Simulated Annealing)")
print(f"Critical time tc = {tc:.2f} days after start (date: {dates[0] + pd.Timedelta(days=int(tc))})")
print(f"Exponent m       = {m:.6f} (must be 0<m<1) -> {'OK' if 0<m<1 else 'ERROR'}")
print(f"Angular freq ω   = {omega:.6f} (positive)  -> {'OK' if omega>0 else 'ERROR'}")
print(f"Trend coeff B    = {B:.6f} (should be <0) -> {'OK' if B<0 else 'violated'}")
print(f"Amplitude (C1,C2)= {amplitude:.6f} (should be <1) -> {'OK' if amplitude<1 else 'violated'}")
print(f"A                = {A:.6f}")
print(f"C1               = {C1:.6f}")
print(f"C2               = {C2:.6f}")
print(f"Phase φ          = {phi:.6f} rad")
print(f"Final RSS        = {best_rss:.4f}")
print(f"RMSE             = {np.sqrt(best_rss/len(y)):.4f}")

# =========================
# 8. PLOT WITH DATES ON X-AXIS
# =========================
t_fit = np.linspace(t[0], tc, 300)
X_fit = build_design_matrix(t_fit, tc, m, omega)
log_price_fit = X_fit @ [A, B, C1, C2]
price_fit = np.exp(log_price_fit)
fit_dates = dates[0] + pd.to_timedelta(t_fit, unit='D')

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(dates, y, 'bo', markersize=3, label='Observed log price')
plt.plot(fit_dates, log_price_fit, 'r-', linewidth=2, label='LPPL fit')
tc_date = dates[0] + pd.Timedelta(days=int(tc))
plt.axvline(x=tc_date, color='k', linestyle='--', label=f'tc = {tc_date.strftime("%Y-%m-%d")}')
plt.xlabel('Date')
plt.ylabel('log(Price)')
plt.title('JLS Model Fit (Log Scale)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

plt.subplot(1,2,2)
plt.plot(dates, prices, 'bo', markersize=3, label='Actual price')
plt.plot(fit_dates, price_fit, 'r-', linewidth=2, label='LPPL fit')
plt.axvline(x=tc_date, color='k', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Price Fit (Original Scale)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Convergence plot
plt.figure(figsize=(10,4))
plt.plot(hist)
plt.title('Convergence of RSS (Ordinary Simulated Annealing)')
plt.xlabel('Iteration')
plt.ylabel('RSS')
plt.grid(True)
plt.show()