#!/usr/bin/env python3
"""Receding-horizon contact MINLP with two simultaneous planar pushers.

At each prediction stage the left and right contact binaries are independent,
so the four modes are free, left-only, right-only, and both-active.  Discrete
modes are enumerated; each fixed-mode nonlinear force problem uses SLSQP.
"""
import argparse
from itertools import product
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def step(x, f, dt=.1, mass=1., inertia=.05, half=.15):
    """Nonlinear semi-implicit step; f=[fnL,ftL,fnR,ftR]."""
    R = rot(x[2]); tangent = R @ np.array([0., 1.])
    nL, nR = R @ np.array([-1., 0.]), R @ np.array([1., 0.])
    armL, armR = half*nL, half*nR
    forceL = -f[0]*nL + f[1]*tangent
    forceR = -f[2]*nR + f[3]*tangent
    force = forceL + forceR
    torque = np.cross(np.r_[armL, 0.], np.r_[forceL, 0.])[2] + np.cross(np.r_[armR, 0.], np.r_[forceR, 0.])[2]
    v = x[3:5] + dt*force/mass
    w = x[5] + dt*torque/inertia
    return np.r_[x[:2] + dt*v, x[2] + dt*w, v, w]


def rollout(x0, forces, dt):
    xs = [np.asarray(x0, float)]
    for f in forces: xs.append(step(xs[-1], f, dt))
    return np.asarray(xs)


def solve(x0, goal, horizon, dt, force_max=8., mu=.5):
    """Enumerate (zL,zR) in {0,1}² over the horizon and solve each NLP."""
    best = None
    for bits in product((0, 1), repeat=2*horizon):
        modes = np.asarray(bits, int).reshape(horizon, 2)
        bounds = []
        for zL, zR in modes:
            bounds += [(0, force_max) if zL else (0, 0), (-force_max, force_max) if zL else (0, 0)]
            bounds += [(0, force_max) if zR else (0, 0), (-force_max, force_max) if zR else (0, 0)]
        guess = np.zeros((horizon, 4)); guess[:, 0] = 2*modes[:, 0]; guess[:, 2] = 2*modes[:, 1]
        def states(v): return rollout(x0, v.reshape(horizon, 4), dt)
        def cost(v):
            xs, fs = states(v), v.reshape(horizon, 4)
            e = xs[-1] - goal
            return e @ np.diag([250, 250, 40, 10, 10, 2]) @ e + .02*np.sum(fs**2)
        constraints = []
        for k in range(horizon):
            constraints += [{"type":"ineq", "fun":lambda v,k=k: mu*v.reshape(horizon,4)[k,0]-abs(v.reshape(horizon,4)[k,1])},
                            {"type":"ineq", "fun":lambda v,k=k: mu*v.reshape(horizon,4)[k,2]-abs(v.reshape(horizon,4)[k,3])}]
        r = minimize(cost, guess.ravel(), method="SLSQP", bounds=bounds, constraints=constraints,
                     options={"maxiter":400, "ftol":1e-8})
        if r.success and (best is None or r.fun < best["cost"]):
            best = {"cost":float(r.fun), "modes":modes, "forces":r.x.reshape(horizon,4), "states":states(r.x)}
    if best is None: raise RuntimeError("no feasible two-pusher mode")
    return best


def contact_positions(x, half=.15, radius=.03):
    R=rot(x[2]); return np.array([x[:2]+R@[-half-radius,0], x[:2]+R@[half+radius,0]])


def run(steps=3, horizon=3, dt=.1):
    x=np.zeros(6); goal=np.array([.18,.02,.12,0,0,0]); xs=[x.copy()]; ps=[contact_positions(x)]; fs=[]; modes=[]
    for k in range(steps):
        plan=solve(x,goal,horizon,dt); f, m=plan["forces"][0],plan["modes"][0]
        x=step(x,f,dt); xs.append(x.copy()); ps.append(contact_positions(x)); fs.append(f); modes.append(m)
        print(f"step {k+1}/{steps}: zL,zR={m.tolist()}, force={f}")
    return np.asarray(xs),np.asarray(ps),np.asarray(fs),np.asarray(modes),goal


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--steps",type=int,default=3); p.add_argument("--horizon",type=int,default=3)
    p.add_argument("--output",type=Path,default=ROOT/"planar_pushing/outputs/two_pusher_minlp.npz"); a=p.parse_args()
    xs,ps,fs,modes,goal=run(a.steps,a.horizon); a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,time=np.arange(len(xs))*.1,states=xs,pusher_positions=ps,contact_forces=fs,contact_mode=modes,goal_state=goal,timestep=np.array(.1))
    print("final:",xs[-1]); print("saved:",a.output)
if __name__=="__main__": main()
