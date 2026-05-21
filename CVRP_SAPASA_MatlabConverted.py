from google.colab import files
import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy
import math

# =========================
# 1. LOAD FILE (unchanged)
# =========================
uploaded = files.upload()
filename = list(uploaded.keys())[0]

with open(filename) as f:
    lines = f.readlines()

coords, demand = {}, {}
capacity = None
depot = None
mode = None

for line in lines:
    line = line.strip()
    if "CAPACITY" in line:
        capacity = int(line.split(":")[1])
    if "NODE_COORD_SECTION" in line:
        mode = "coord"
        continue
    if "DEMAND_SECTION" in line:
        mode = "dem"
        continue
    if "DEPOT_SECTION" in line:
        mode = "dep"
        continue
    if line == "EOF":
        break

    if mode == "coord":
        i, x, y = line.split()
        coords[int(i)] = (float(x), float(y))
    if mode == "dem":
        i, d = line.split()
        demand[int(i)] = float(d)
    if mode == "dep":
        v = int(line)
        if v != -1:
            depot = v

# =========================
# 2. DATA PREPARATION (unchanged)
# =========================
nodes = sorted(coords.keys())
nodes.remove(depot)
nodes = [depot] + nodes

x = np.array([coords[i][0] for i in nodes])
y = np.array([coords[i][1] for i in nodes])
r = np.array([demand[i] for i in nodes])

N = len(nodes)
J = 5   # vehicles (must match instance)

dist = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        dist[i, j] = np.hypot(x[i]-x[j], y[i]-y[j])

d0 = np.array([dist[i, 0] for i in range(N)])

customers = list(range(1, N))

# =========================
# 3. HELPER FUNCTIONS (unchanged)
# =========================
def route_load(route):
    return sum(r[c] for c in route)

def route_cost(route):
    if not route:
        return 0
    cost = d0[route[0]]
    for i in range(len(route)-1):
        cost += dist[route[i], route[i+1]]
    cost += d0[route[-1]]
    return cost

def total_cost(routes):
    return sum(route_cost(r) for r in routes)

def is_feasible(routes):
    return all(route_load(rt) <= capacity for rt in routes)

# =========================
# 4. INITIAL SOLUTION (unchanged)
# =========================
def clarke_wright():
    routes = [[c] for c in customers]
    loads = [r[c] for c in customers]
    savings = []
    for i in range(len(customers)):
        for j in range(i+1, len(customers)):
            ci, cj = customers[i], customers[j]
            save = dist[0, ci] + dist[0, cj] - dist[ci, cj]
            savings.append((save, ci, cj))
    savings.sort(reverse=True, key=lambda x: x[0])
    cust_to_route = {c: i for i, rt in enumerate(routes) for c in rt}
    for save, ci, cj in savings:
        if cust_to_route[ci] == cust_to_route[cj]:
            continue
        ri = cust_to_route[ci]
        rj = cust_to_route[cj]
        if loads[ri] + loads[rj] <= capacity:
            routes[ri].extend(routes[rj])
            loads[ri] += loads[rj]
            for c in routes[rj]:
                cust_to_route[c] = ri
            routes[rj] = []
            loads[rj] = 0
    routes = [rt for rt in routes if rt]
    while len(routes) < J:
        routes.append([])
    return routes

# =========================
# 5. INTRA-ROUTE 2-OPT (unchanged)
# =========================
def two_opt(route):
    if len(route) <= 3:
        return route
    improved = True
    while improved:
        improved = False
        n = len(route)
        best_gain = 0
        best_i, best_j = -1, -1
        for i in range(n-1):
            for j in range(i+2, n):
                a, b = route[i], route[i+1]
                c = route[j]
                if j+1 < n:
                    nxt = route[j+1]
                    old = dist[a,b] + dist[c,nxt]
                    new = dist[a,c] + dist[b,nxt]
                else:
                    old = dist[a,b] + d0[c]
                    new = dist[a,c] + d0[b]
                gain = new - old
                if gain < best_gain:
                    best_gain = gain
                    best_i, best_j = i, j
        if best_gain < 0:
            route[best_i+1:best_j+1] = reversed(route[best_i+1:best_j+1])
            improved = True
    return route

# =========================
# 6. INTER-ROUTE RELOCATE (unchanged)
# =========================
def best_relocate(routes):
    best_gain = 0
    best_move = None
    for a in range(J):
        for idx in range(len(routes[a])):
            cust = routes[a][idx]
            for b in range(J):
                if a == b: continue
                if route_load(routes[b]) + r[cust] <= capacity:
                    old_cost_a = route_cost(routes[a])
                    old_cost_b = route_cost(routes[b])
                    temp_a = routes[a][:idx] + routes[a][idx+1:]
                    new_cost_a = route_cost(temp_a)
                    best_inc = float('inf')
                    for pos in range(len(routes[b])+1):
                        temp_b = routes[b][:pos] + [cust] + routes[b][pos:]
                        inc = route_cost(temp_b) - old_cost_b
                        if inc < best_inc:
                            best_inc = inc
                    gain = (new_cost_a + old_cost_b + best_inc) - (old_cost_a + old_cost_b)
                    if gain < best_gain:
                        best_gain = gain
                        best_move = (a, idx, b)
    if best_gain < 0:
        a, idx, b = best_move
        cust = routes[a].pop(idx)
        best_pos = 0
        best_inc = float('inf')
        for pos in range(len(routes[b])+1):
            if pos == 0:
                inc = d0[cust] + (dist[cust, routes[b][0]] if routes[b] else 0) - (d0[routes[b][0]] if routes[b] else 0)
            elif pos == len(routes[b]):
                inc = dist[routes[b][-1], cust] + d0[cust] - d0[routes[b][-1]]
            else:
                inc = dist[routes[b][pos-1], cust] + dist[cust, routes[b][pos]] - dist[routes[b][pos-1], routes[b][pos]]
            if inc < best_inc:
                best_inc = inc
                best_pos = pos
        routes[b].insert(best_pos, cust)
        return True
    return False

# =========================
# 7. INTER-ROUTE 2-OPT* (unchanged)
# =========================
def two_opt_star(routes):
    improved = False
    for a in range(J):
        for b in range(a+1, J):
            r1, r2 = routes[a], routes[b]
            if not r1 or not r2:
                continue
            best_gain = 0
            best_i, best_j = -1, -1
            for i in range(len(r1)):
                for j in range(len(r2)):
                    new_r1 = r1[:i+1] + r2[j+1:]
                    new_r2 = r2[:j+1] + r1[i+1:]
                    if route_load(new_r1) <= capacity and route_load(new_r2) <= capacity:
                        gain = (route_cost(new_r1) + route_cost(new_r2)) - (route_cost(r1) + route_cost(r2))
                        if gain < best_gain:
                            best_gain = gain
                            best_i, best_j = i, j
            if best_gain < 0:
                i, j = best_i, best_j
                routes[a] = r1[:i+1] + r2[j+1:]
                routes[b] = r2[:j+1] + r1[i+1:]
                improved = True
    return routes, improved

# =========================
# 8. LOCAL SEARCH (unchanged)
# =========================
def local_search(routes):
    for i in range(J):
        routes[i] = two_opt(routes[i][:])
    improved = True
    while improved:
        improved = False
        if best_relocate(routes):
            improved = True
        routes, imp = two_opt_star(routes)
        if imp:
            improved = True
    return routes

# =========================
# 9. RUIN-AND-RECREATE (strictly feasible, unchanged)
# =========================
def ruin_and_recreate(routes, ruin_frac=0.3):
    new = [r[:] for r in routes]
    all_cust = [c for rt in new for c in rt]
    if not all_cust:
        return new
    n_ruin = max(1, int(len(all_cust) * ruin_frac))
    to_remove = random.sample(all_cust, n_ruin)
    for rt in new:
        rt[:] = [c for c in rt if c not in to_remove]
    random.shuffle(to_remove)
    for cust in to_remove:
        best_cost_inc = float('inf')
        best_route = None
        best_pos = None
        for j in range(J):
            if route_load(new[j]) + r[cust] <= capacity:
                rt = new[j]
                for pos in range(len(rt)+1):
                    if pos == 0:
                        inc = d0[cust] + (dist[cust, rt[0]] if rt else 0) - (d0[rt[0]] if rt else 0)
                    elif pos == len(rt):
                        inc = dist[rt[-1], cust] + d0[cust] - d0[rt[-1]]
                    else:
                        inc = dist[rt[pos-1], cust] + dist[cust, rt[pos]] - dist[rt[pos-1], rt[pos]]
                    if inc < best_cost_inc:
                        best_cost_inc = inc
                        best_route = j
                        best_pos = pos
        if best_route is None:
            return ruin_and_recreate(routes, ruin_frac * 0.9)
        new[best_route].insert(best_pos, cust)
    return new

# =========================
# 10. WEIBULL PARAMETER ESTIMATION (new)
# =========================
def estimate_weibull_params(errors):
    """Given a list of positive numbers, estimate (shape k, scale λ) using moment matching."""
    if len(errors) < 5:
        return 1.0, np.mean(errors) if errors else 1.0
    mean = np.mean(errors)
    std = np.std(errors)
    cv = std / mean if mean > 0 else 1.0
    # Coefficient of variation for Weibull: sqrt( Γ(1+2/k) / Γ(1+1/k)^2 - 1 )
    # Solve for k using binary search
    if cv < 0.01:
        k = 10.0  # nearly deterministic
    else:
        lo, hi = 0.1, 20.0
        for _ in range(50):
            mid = (lo + hi) / 2
            g1 = math.gamma(1 + 1/mid)
            g2 = math.gamma(1 + 2/mid)
            weibull_cv = math.sqrt(g2 / (g1*g1) - 1)
            if weibull_cv < cv:
                hi = mid
            else:
                lo = mid
        k = (lo + hi) / 2
    # Compute scale λ = mean / Γ(1+1/k)
    lam = mean / math.gamma(1 + 1/k)
    return k, lam

# =========================
# 11. ITERATED LOCAL SEARCH WITH WEIBULL‑ADAPTIVE SA (revised)
# =========================
def solve(iterations=4000, restart_after=800, warmup_iterations=100):
    # initial feasible solution
    best = clarke_wright()
    best = local_search(best)
    best_cost = total_cost(best)
    print(f"Initial best: {best_cost:.2f}")
    
    current = deepcopy(best)
    current_cost = best_cost
    
    # SA parameters
    T0 = 100.0          # initial temperature
    alpha = 0.995       # cooling rate
    T = T0
    
    # Collect errors (Δ = candidate_cost - current_cost) for worse solutions
    near_errors = []
    distant_errors = []
    
    history = [best_cost]
    no_improve = 0
    
    for it in range(iterations):
        # Generate candidate (feasible)
        if random.random() < 0.7:
            candidate = ruin_and_recreate(current, ruin_frac=0.2 + 0.1 * random.random())
        else:
            candidate = [r[:] for r in current]
            a, b = random.sample(range(J), 2)
            if candidate[a] and candidate[b]:
                ia = random.randrange(len(candidate[a]))
                ib = random.randrange(len(candidate[b]))
                new_load_a = route_load(candidate[a]) - r[candidate[a][ia]] + r[candidate[b][ib]]
                new_load_b = route_load(candidate[b]) - r[candidate[b][ib]] + r[candidate[a][ia]]
                if new_load_a <= capacity and new_load_b <= capacity:
                    candidate[a][ia], candidate[b][ib] = candidate[b][ib], candidate[a][ia]
        
        candidate = local_search(candidate)
        if not is_feasible(candidate):
            continue
        candidate_cost = total_cost(candidate)
        delta = candidate_cost - current_cost
        
        # Acceptance decision
        accept = False
        if delta < 0:
            accept = True  # improvement
        else:
            # Worse solution – decide using adaptive Weibull or standard SA during warm‑up
            if it < warmup_iterations:
                # Standard Metropolis
                if random.random() < math.exp(-delta / T):
                    accept = True
                # Record error for later parameter estimation
                if delta > 0:
                    # Classify temporarily using a simple threshold (relative to current cost)
                    if delta < 0.05 * current_cost:
                        near_errors.append(delta)
                    else:
                        distant_errors.append(delta)
            else:
                # Adaptive Weibull acceptance
                # Classify the error as near or distant
                if delta < 0.05 * current_cost:
                    cat = "near"
                    # Estimate Weibull parameters for near errors (if enough data)
                    if len(near_errors) >= 5:
                        k, lam = estimate_weibull_params(near_errors)
                    else:
                        k, lam = 1.0, np.mean(near_errors) if near_errors else 1.0
                    # Effective temperature: slower cooling (log)
                    T_eff = math.log(1 + T)
                else:
                    cat = "distant"
                    if len(distant_errors) >= 5:
                        k, lam = estimate_weibull_params(distant_errors)
                    else:
                        k, lam = 1.0, np.mean(distant_errors) if distant_errors else 1.0
                    # Effective temperature: faster cooling (T)
                    T_eff = T
                
                # Weibull acceptance probability: exp( - (Δ / (λ * T_eff))^k )
                if lam > 0 and T_eff > 0:
                    prob = math.exp(- ((delta / (lam * T_eff)) ** k))
                else:
                    prob = 0.0
                if random.random() < prob:
                    accept = True
                # Store error for future estimation (regardless of acceptance)
                if cat == "near":
                    near_errors.append(delta)
                else:
                    distant_errors.append(delta)
                # Trim error lists to keep only recent 500 samples
                if len(near_errors) > 500:
                    near_errors = near_errors[-500:]
                if len(distant_errors) > 500:
                    distant_errors = distant_errors[-500:]
        
        if accept:
            current = deepcopy(candidate)
            current_cost = candidate_cost
            if candidate_cost < best_cost:
                best = deepcopy(candidate)
                best_cost = candidate_cost
                no_improve = 0
                print(f"Iter {it:5d} | T={T:.2f} | new best = {best_cost:.2f}")
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # Cooling schedule (geometric)
        T *= alpha
        # Keep temperature above a small positive value
        if T < 0.01:
            T = 0.01
        
        history.append(best_cost)
        
        # Restart if stagnation
        if no_improve > restart_after:
            print(f"Restart at iteration {it} (no improvement for {restart_after})")
            current = ruin_and_recreate(best, ruin_frac=0.5)
            current = local_search(current)
            current_cost = total_cost(current)
            no_improve = 0
            # Reset temperature after restart
            T = T0
            # Optionally clear error lists to re-adapt (keep for now)
    
    return best, best_cost, history

# =========================
# 12. RUN
# =========================
best_routes, best_cost, hist = solve(iterations=4000, restart_after=800, warmup_iterations=100)

print(f"\n{'='*50}")
print(f"✅ FINAL COST: {best_cost:.2f}")
print(f"Optimal for A-n32-k5 is 784.0")
if best_cost <= 784:
    print("You found the optimum (or better) – but better is impossible; check feasibility.")
else:
    print(f"Feasible solution, cost = {best_cost:.2f}")
print("Routes (depot omitted):")
for j, rt in enumerate(best_routes):
    if rt:
        load = route_load(rt)
        print(f"V{j}: {rt}  load={load:.0f}/{capacity}  cost={route_cost(rt):.2f}")

# Convergence plot
plt.figure(figsize=(10,4))
plt.plot(hist)
plt.axhline(y=784, color='r', linestyle='--', label='Optimal (784)')
plt.title("ILS with Weibull‑Adaptive SA (strictly feasible)")
plt.grid()
plt.legend()
plt.show()

# Plot routes
plt.figure(figsize=(10,8))
for rt in best_routes:
    if not rt:
        continue
    X = [x[0]] + [x[i] for i in rt] + [x[0]]
    Y = [y[0]] + [y[i] for i in rt] + [y[0]]
    plt.plot(X, Y, marker='o', linewidth=1.5)
plt.scatter([x[0]],[y[0]], c='red', s=150, label='Depot')
plt.title("CVRP Solution (feasible, near-optimal)")
plt.legend()
plt.grid()
plt.show()