import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)
os.makedirs("collision_gifs2", exist_ok=True)


def finite_diff(y, t):
    # y: [N, 2], t: [N,1]
    dt = (t[1:] - t[:-1]).clamp_min(1e-8)  # [N-1,1]
    v = (y[1:] - y[:-1]) / dt              # [N-1,2]
    return v


# def generate_observed_trajectory(m1_true, m2_true, v1_0=2.0, v2_0=0.0, T=2.0, N=400):
#     # Geometry for visualization only
#     base_radius = 0.08
#     r1_true = base_radius * (m1_true ** (1/3))
#     r2_true = base_radius * (m2_true ** (1/3))
#     R_true = r1_true + r2_true

#     # Initial positions
#     x1_0, x2_0 = -1.0, +1.0

#     t = torch.linspace(0., T, N)
#     tc = ((x2_0 - x1_0) - R_true) / (v1_0 - v2_0 + 1e-8)

#     x1 = torch.empty_like(t)
#     x2 = torch.empty_like(t)
#     pre = t <= tc
#     x1[pre] = x1_0 + v1_0 * t[pre]
#     x2[pre] = x2_0 + v2_0 * t[pre]

#     v1p = ((m1_true - m2_true)*v1_0 + 2*m2_true*v2_0) / (m1_true + m2_true)
#     v2p = ((m2_true - m1_true)*v2_0 + 2*m1_true*v1_0) / (m1_true + m2_true)

#     x1_tc = x1_0 + v1_0 * tc
#     x2_tc = x2_0 + v2_0 * tc
#     post = t > tc
#     dt_post = t[post] - tc
#     x1[post] = x1_tc + v1p * dt_post
#     x2[post] = x2_tc + v2p * dt_post

#     x_data = torch.stack([x1, x2], dim=1)  # [N,2]
#     return t.unsqueeze(1), x_data, r1_true, r2_true, (x1_0, x2_0)

def generate_observed_trajectory(m1_true, m2_true, v1_0=2.0, v2_0=0.0, T=2.0, N=400, R_dyn=0.2):
    base_radius = 0.08
    r1_true = base_radius * (m1_true ** (1/3))
    r2_true = base_radius * (m2_true ** (1/3))
    x1_0, x2_0 = -1.0, +1.0
    t = torch.linspace(0., T, N)
    rel_speed = (v1_0 - v2_0) + 1e-8
    tc_val = ((x2_0 - x1_0) - R_dyn) / rel_speed
    tc = t.new_tensor(tc_val)
    tc = torch.clamp(tc, min=t[1], max=t[-2])
    x1 = torch.empty_like(t); x2 = torch.empty_like(t)
    pre = t <= tc
    x1[pre] = x1_0 + v1_0 * t[pre]
    x2[pre] = x2_0 + v2_0 * t[pre]
    v1p = ((m1_true - m2_true)*v1_0 + 2*m2_true*v2_0) / (m1_true + m2_true)
    v2p = ((m2_true - m1_true)*v2_0 + 2*m1_true*v1_0) / (m1_true + m2_true)
    x1_tc = x1_0 + v1_0 * tc
    x2_tc = x2_0 + v2_0 * tc
    post = t > tc
    dt_post = t[post] - tc
    x1[post] = x1_tc + v1p * dt_post
    x2[post] = x2_tc + v2p * dt_post

    x_data = torch.stack([x1, x2], dim=1)  # [N,2]
    return t.unsqueeze(1), x_data, r1_true, r2_true, (x1_0, x2_0)



class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(1, 64), nn.Tanh()]
        for _ in range(2):
            layers += [nn.Linear(64, 64), nn.Tanh()]
        layers += [nn.Linear(64, 2)]
        self.net = nn.Sequential(*layers)
    def forward(self, t): 
        return self.net(t)


def run_candidate(m1, m2, t, x_data, steps=5000, lr=1e-3, return_artifacts=False):
    t = t.to(device)
    x_data = x_data.to(device)

    net = MLP().to(device)
    opt = optim.Adam(net.parameters(), lr=lr)

    loss_total_list, loss_data_list, loss_phys_list = [], [], []

    for it in range(steps):
        opt.zero_grad()
        x_pred = net(t)  # [N,2]

        loss_data = ((x_pred - x_data) ** 2).mean()
        gap = x_pred[:, 1] - x_pred[:, 0]          # [N]
        tc_idx = int(torch.argmin(gap).item())    
        v_fd = finite_diff(x_pred, t)              # [N-1,2]
        pre_i  = max(0, min(v_fd.shape[0]-1, tc_idx-1))
        post_i = max(0, min(v_fd.shape[0]-1, tc_idx))
        v_minus = v_fd[pre_i]      # [2] velocities just before collision
        v_plus  = v_fd[post_i]     # [2] velocities just after collision
        mom_before = m1 * v_minus[0] + m2 * v_minus[1]
        mom_after  = m1 * v_plus[0]  + m2 * v_plus[1]
        mom_resid  = (mom_after - mom_before)
        en_before = 0.5 * m1 * (v_minus[0]**2) + 0.5 * m2 * (v_minus[1]**2)
        en_after  = 0.5 * m1 * (v_plus[0]**2)  + 0.5 * m2 * (v_plus[1]**2)
        en_resid  = (en_after - en_before)
        loss_gov = mom_resid.pow(2) + en_resid.pow(2)
        loss = loss_data + loss_gov
        loss.backward()
        opt.step()

        loss_total_list.append(loss.item())
        loss_data_list.append(loss_data.item())
        loss_phys_list.append(loss_gov.item())

    net.eval()
    with torch.no_grad():
        x_pred = net(t)
        mse = ((x_pred - x_data)**2).mean().item()
        gap = x_pred[:, 1] - x_pred[:, 0]
        tc_idx = int(torch.argmin(gap).item())
        v_fd = finite_diff(x_pred, t)
        pre_i  = max(0, min(v_fd.shape[0]-1, tc_idx-1))
        post_i = max(0, min(v_fd.shape[0]-1, tc_idx))
        v_minus = v_fd[pre_i]
        v_plus  = v_fd[post_i]
        mom_resid = (m1 * v_plus[0] + m2 * v_plus[1]) - (m1 * v_minus[0] + m2 * v_minus[1])
        en_resid  = (0.5*m1*v_plus[0]**2 + 0.5*m2*v_plus[1]**2) - (0.5*m1*v_minus[0]**2 + 0.5*m2*v_minus[1]**2)
        loss_gov = (mom_resid**2 + en_resid**2).item()
        total = mse + loss_gov

    artifacts = {
        "x_pred": x_pred.detach().cpu().numpy(),
        "x_true": x_data.detach().cpu().numpy(),
        "losses": {"total": loss_total_list, "data": loss_data_list, "gov": loss_phys_list},
        "t": t.detach().cpu().squeeze().numpy(),
        "rank_score": total,
        "mse": mse,
        "gov": loss_gov
    }
    return total, net, artifacts


# def animate(x_vals, t_vals, r1, r2, filename, title):
#     fig, ax = plt.subplots(figsize=(6,2))
#     ax.set_aspect('equal', 'box')
#     ax.set_xlim(x_vals.min() - 0.2, x_vals.max() + 0.2)
#     ax.set_ylim(-0.2, 0.2)
#     ax.set_xlabel('x')
#     time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)
#     ax.set_title(title)
#     c1 = Circle((x_vals[0,0], 0.0), r1, fc='C0', ec='k')
#     c2 = Circle((x_vals[0,1], 0.0), r2, fc='C1', ec='k')
#     ax.add_patch(c1); ax.add_patch(c2)
#     def update(i):
#         c1.center = (x_vals[i,0], 0.0)
#         c2.center = (x_vals[i,1], 0.0)
#         time_text.set_text(f't = {t_vals[i]:.3f}s')
#         return c1, c2, time_text
#     ani = FuncAnimation(fig, update, frames=range(0, len(t_vals), 4), interval=10, blit=True)
#     ani.save(filename, writer=PillowWriter(fps=60))
#     plt.close()

def animate(x_vals, t_vals, r1, r2, filename, title):
    import numpy as np

    x_vis = np.array(x_vals, dtype=float).copy()
    R = float(r1 + r2)

    gap = x_vis[:, 1] - x_vis[:, 0]
    mask = gap < R
    if np.any(mask):
        mid  = 0.5 * (x_vis[mask, 0] + x_vis[mask, 1])
        need = 0.5 * (R - gap[mask])
        x_vis[mask, 0] = mid - need
        x_vis[mask, 1] = mid + need

    eps = 5e-4 * (x_vis.max() - x_vis.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(x_vis.min() - 0.2, x_vis.max() + 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_xlabel('x')
    time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)
    ax.set_title(title)

    c1 = Circle((x_vis[0, 0], 0.0), r1, fc='C0', ec='k', zorder=2)
    c2 = Circle((x_vis[0, 1], 0.0), r2, fc='C1', ec='k', zorder=2)
    ax.add_patch(c1); ax.add_patch(c2)

    def update(i):
        x1i, x2i = x_vis[i, 0], x_vis[i, 1]
        gi = x2i - x1i
        if gi < R + eps:
            mid = 0.5 * (x1i + x2i)
            need = 0.5 * ((R + eps) - gi)
            x1i = mid - need
            x2i = mid + need
        c1.center = (x1i, 0.0)
        c2.center = (x2i, 0.0)
        time_text.set_text(f't = {t_vals[i]:.3f}s')
        return c1, c2, time_text

    ani = FuncAnimation(fig, update, frames=range(0, len(t_vals), 4), interval=10, blit=True)
    ani.save(filename, writer=PillowWriter(fps=60))
    plt.close()



def run_mass_sweep(candidate_masses, true_masses, v1_0=2.0, v2_0=0.0, steps=10000, tag="run"):
    m1_true, m2_true = true_masses
    t, x_obs, r1_true, r2_true, (x1_0, x2_0) = generate_observed_trajectory(
        m1_true, m2_true, v1_0=v1_0, v2_0=v2_0
    )
    t = t.to(device); x_obs = x_obs.to(device)

    best = {"m1": None, "m2": None, "score": float("inf"), "net": None, "art": None}

    for (m1, m2) in candidate_masses:
        score, net, art = run_candidate(m1, m2, t, x_obs, steps=steps, return_artifacts=True)
        print(f"[{tag}] Candidate (m1={m1}, m2={m2}) → total={score:.6e} | data={art['mse']:.6e} | gov={art['gov']:.6e}")
        if score < best["score"]:
            best.update({"m1": m1, "m2": m2, "score": score, "net": net, "art": art})

    base_radius = 0.08
    r1_cand = base_radius * (best["m1"] ** (1/3))
    r2_cand = base_radius * (best["m2"] ** (1/3))

    x_pred = best["art"]["x_pred"]
    x_true = best["art"]["x_true"]
    t_vals = best["art"]["t"]

    tag_best = f"true_{m1_true}_{m2_true}__pred_{best['m1']}_{best['m2']}_v1_{v1_0}_v2_{v2_0}"
    animate(x_pred, t_vals, r1_cand, r2_cand,
            f"collision_gifs2/pred_{tag_best}.gif",
            f"PINN Prediction (true: {m1_true},{m2_true} | pred: {best['m1']},{best['m2']})")
    animate(x_true, t_vals, r1_true, r2_true,
            f"collision_gifs2/true_{tag_best}.gif",
            f"Ground Truth (true: {m1_true},{m2_true})")

    L = best["art"]["losses"]
    plt.figure()
    plt.plot(L["total"], label='Total (data + governing)')
    plt.plot(L["data"], label='Data MSE')
    plt.plot(L["gov"],  label='Governing residual')
    plt.yscale('log'); plt.xlabel("Iteration"); plt.ylabel("Loss")
    plt.legend(); plt.title(f"Loss Curve (true {m1_true},{m2_true} | pred {best['m1']},{best['m2']})"); plt.tight_layout()
    plt.savefig(f"collision_gifs2/loss_{tag_best}.png"); plt.close()

    print(f"\n=== BEST MASS PAIR for {tag} ===")
    print(f"True masses: (m1={m1_true}, m2={m2_true})")
    print(f"Best candidate: (m1={best['m1']}, m2={best['m2']}) with TOTAL={best['score']:.6e}")
    return best


# candidate_masses = [
#     (1.0, 1.0), (2.0, 1.0), (2.0, 10.0), (10, 1.0), (1.0, 2.0),
#     (3.0, 1.0), (1.0, 3.0), (2.5, 2.0), (4.0, 1.0), (1.0, 4.0)
# ]

test_configs = [
    (1.0, 1.0, 2.0, 0.0),     
    (2.0, 1.0, 3.0, 0.0),     
    (2.0, 10.0, 4.0, 0.0),    
    (2.0, 3.0, 4.0, 0.0),   
    (20.0, 10.0, 5.0, 0.0),  
    (1.0, 3.0, 5.0, 0.0),
    (5.0, 1.0, 1.0, 0.0),
    (65.0, 5.0, 6.0, 0.0),
]


# more_test_configs = [
#     (1.0, 1.0, 3.0, -1.0),
#     (5.0, 5.0, 2.0, -2.0),
#     (10.0, 10.0, 1.5, -0.5),
#     (1.0, 5.0, 4.0, 0.0),
#     (1.0, 10.0, 4.5, 0.0),
#     (0.5, 5.0, 5.0, 0.0),
#     (10.0, 1.0, 3.0, 0.0),
#     (5.0, 1.0, 4.5, 0.0),
#     (20.0, 1.0, 5.0, 0.0),
#     (2.0, 1.0, 3.0, -1.0),
#     (2.0, 3.0, 4.0, -2.0),
#     (3.0, 2.0, 2.5, -1.5),
#     (2.0, 1.0, 5.0, 2.0),
#     (3.0, 5.0, 6.0, 4.0),
#     (10.0, 2.0, 3.5, 1.0),
#     (1.0, 100.0, 5.0, 0.0),
#     (100.0, 1.0, 2.5, 0.0),
#     (2.0, 2.0, 0.8, 0.0),
#     (3.0, 4.0, 1.0, -0.2),
#     (4.0, 2.0, 2.0, 0.0),
#     (2.0, 4.0, 5.0, 0.0),
#     (7.0, 3.0, 4.0, -1.0),
#     (3.0, 7.0, 4.0, 1.0),
#     (1.0, 1.0, 2.0, -2.0),
# ]

# test_configs = test_configs + more_test_configs



@torch.no_grad()
def extract_vel_features(t, x, k=10):
    v_fd = finite_diff(x, t)           # [N-1,2]
    gap = (x[:,1] - x[:,0])            # [N]
    tc_idx = int(torch.argmin(gap).item())

    tc_idx = max(k, min(tc_idx, v_fd.shape[0]-k))

    v_minus = v_fd[tc_idx-k:tc_idx].mean(dim=0)   # [2]
    v_plus  = v_fd[tc_idx:tc_idx+k].mean(dim=0)   # [2]
    feat = torch.stack([v_minus[0], v_minus[1], v_plus[0], v_plus[1]], dim=0)
    return feat

class MassMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )
        self.softplus = nn.Softplus()
    def forward(self, feat_n):
        logm = self.net(feat_n)           
        m = self.softplus(logm) + 1e-4    
        return m

def train_mass_mlp(num_sims=4000, epochs=200, lr=1e-3, 
                   T=2.0, N=400, v1_range=(0.5,5.0), v2_range=(-0.5,1.0),
                   m_range=(0.5,20.0), device=device, verbose=True):
    model = MassMLP().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    feats, targets = [], []
    for _ in range(num_sims):
        m1 = float(np.exp(np.random.uniform(np.log(m_range[0]), np.log(m_range[1]))))
        m2 = float(np.exp(np.random.uniform(np.log(m_range[0]), np.log(m_range[1]))))
        v1_0 = float(np.random.uniform(*v1_range))
        v2_0 = float(np.random.uniform(*v2_range))

        t, x_data, _, _, _ = generate_observed_trajectory(m1, m2, v1_0=v1_0, v2_0=v2_0, T=T, N=N)
        t = t.to(device); x_data = x_data.to(device)
        feat = extract_vel_features(t, x_data)     # [4]
        feats.append(feat)
        targets.append(torch.tensor([m1, m2], device=device))

    X = torch.stack(feats, dim=0).to(device)       # [M,4]
    Y = torch.stack(targets, dim=0).to(device)     # [M,2]

    mu = X.mean(dim=0, keepdim=True)
    sigma = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xn = (X - mu) / sigma
    model.feat_mu = mu.detach()
    model.feat_sigma = sigma.detach()

    model.train()
    bs = 256
    for ep in range(epochs):
        idx = torch.randperm(Xn.shape[0], device=device)
        tot = 0.0
        for i in range(0, Xn.shape[0], bs):
            xb = Xn[idx[i:i+bs]]
            yb = Y[idx[i:i+bs]]
            pred = model(xb)
            loss = ((pred - yb)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xb.size(0)
        if verbose and (ep % 20 == 0 or ep == epochs-1):
            print(f"[MassMLP] epoch {ep+1:03d}/{epochs} | MSE={tot/Xn.shape[0]:.4e}")
    model.eval()
    return model

@torch.no_grad()
def mlp_predict_masses(model, t, x):
    """Predict (m1, m2) from positions x(t) using MassMLP."""
    dev = next(model.parameters()).device
    t = t.to(dev); x = x.to(dev)
    feat = extract_vel_features(t, x)                                 # [4]
    feat_n = (feat - model.feat_mu.squeeze(0)) / model.feat_sigma.squeeze(0)
    m = model(feat_n.unsqueeze(0)).squeeze(0)                          # [2]
    return float(m[0].item()), float(m[1].item())

def run_mass_sweep_mlp(mass_mlp, true_masses, v1_0=2.0, v2_0=0.0, steps=10000, tag="run_mlp"):
    m1_true, m2_true = true_masses

    t, x_obs, r1_true, r2_true, (x1_0, x2_0) = generate_observed_trajectory(
        m1_true, m2_true, v1_0=v1_0, v2_0=v2_0
    )
    t_dev = t.to(device); x_obs_dev = x_obs.to(device)

    m1_hat, m2_hat = mlp_predict_masses(mass_mlp, t_dev, x_obs_dev)
    print(f"[{tag}] MassMLP predicted masses: (m1≈{m1_hat:.3f}, m2≈{m2_hat:.3f})")

    score, net, art = run_candidate(m1_hat, m2_hat, t_dev, x_obs_dev, steps=steps, return_artifacts=True)
    print(f"[{tag}] total={score:.6e} | data={art['mse']:.6e} | gov={art['gov']:.6e}")

    base_radius = 0.08
    r1_cand = base_radius * (m1_hat ** (1/3))
    r2_cand = base_radius * (m2_hat ** (1/3))

    x_pred = art["x_pred"]; x_true = art["x_true"]; t_vals = art["t"]
    tag_best = f"true_{m1_true}_{m2_true}__predMLP_{m1_hat:.3f}_{m2_hat:.3f}_v1_{v1_0}_v2_{v2_0}"

    animate(x_pred, t_vals, r1_cand, r2_cand,
            f"collision_gifs2/pred_{tag_best}.gif",
            f"PINN (MLP masses) true: {m1_true},{m2_true} | pred: {m1_hat:.2f},{m2_hat:.2f}")
    animate(x_true, t_vals, r1_true, r2_true,
            f"collision_gifs2/true_{tag_best}.gif",
            f"Ground Truth (true: {m1_true},{m2_true})")


    L = art["losses"]
    plt.figure()
    plt.plot(L["total"], label='Total (data + governing)')
    plt.plot(L["data"], label='Data MSE')
    plt.plot(L["gov"],  label='Governing residual')
    plt.yscale('log'); plt.xlabel("Iteration"); plt.ylabel("Loss")
    plt.legend(); plt.title(f"Loss (true {m1_true},{m2_true} | predMLP {m1_hat:.2f},{m2_hat:.2f})"); plt.tight_layout()
    plt.savefig(f"collision_gifs2/loss_{tag_best}.png"); plt.close()

    print(f"\n=== MLP MASS RESULT for {tag} ===")
    print(f"True: (m1={m1_true}, m2={m2_true}) | Pred: (m1≈{m1_hat:.3f}, m2≈{m2_hat:.3f})")
    return {"m1_hat": m1_hat, "m2_hat": m2_hat, "score": score, "art": art}

mass_mlp = train_mass_mlp(num_sims=4000, epochs=200, lr=1e-3, verbose=True)

for m1_true, m2_true, v1, v2 in test_configs:
    print(f"\n--- MLP replacement for TRUE (m1={m1_true}, m2={m2_true}), v1={v1}, v2={v2} ---")
    _ = run_mass_sweep_mlp(mass_mlp, (m1_true, m2_true), v1_0=v1, v2_0=v2, steps=10000,
                            tag=f"mlp_true_{m1_true}_{m2_true}")

