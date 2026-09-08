#!/usr/bin/env python3
"""Generate CartPole soft-wall MICP training and test data with Gurobi.

This is the script form of ``data_generation.ipynb``.  It writes the default
configuration and the ``train.p`` / ``test.p`` files below ``cartpole/data``.
"""

import argparse
import pickle
from pathlib import Path

import cvxpy as cp
import numpy as np

from cartpole import Cartpole


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_DIR = ROOT / "cartpole"


def default_problem_parameters():
    """Return the fixed system and contact parameters from the notebook."""
    n, m = 4, 3
    horizon = 11
    dh = 0.05
    q = 10.0 * np.eye(n)
    r = np.diag([1.0, 0.0, 0.0])
    length = cart_mass = pole_mass = 1.0
    gravity = 9.81
    kappa, nu = 100.0, 30.0
    wall_distance = 0.5

    x_max = np.array([wall_distance, np.pi / 8, 2, 1])
    x_min = -x_max
    delta_min = np.array([
        -x_max[0] + length * x_min[1] - wall_distance,
        x_min[0] - length * x_max[1] - wall_distance,
    ])
    ddelta_min = np.array([
        -x_max[2] + length * x_min[3],
        x_min[2] - length * x_max[3],
    ])
    delta_max = np.array([
        -x_min[0] + length * x_max[1] - wall_distance,
        x_max[0] - length * x_min[1] - wall_distance,
    ])
    ddelta_max = np.array([
        -x_min[2] + length * x_max[3],
        x_max[2] - length * x_min[3],
    ])

    uc_min, uc_max = -2.0, 2.0
    sc_min = kappa * delta_min + nu * ddelta_min
    sc_max = kappa * delta_max + nu * ddelta_max

    a = np.zeros((n, n))
    a[: n // 2, n // 2 :] = np.eye(n // 2)
    a[2, 1] = gravity * pole_mass / cart_mass
    a[3, 1] = gravity * (cart_mass + pole_mass) / (length * cart_mass)
    ak = np.eye(n) + dh * a

    b = np.zeros((n, m))
    b[2, 0] = 1 / cart_mass
    b[3, :] = np.array([
        1 / (length * cart_mass),
        -1 / (length * pole_mass),
        1 / (length * pole_mass),
    ])
    bk = dh * b

    return [
        horizon, ak, bk, q, r, x_min, x_max, uc_min, uc_max, sc_min, sc_max,
        delta_min, delta_max, ddelta_min, ddelta_max, dh, gravity, length,
        cart_mass, pole_mass, kappa, nu, wall_distance,
    ]


def write_config(dataset_name, problem_parameters):
    config_path = SYSTEM_DIR / "config" / f"{dataset_name}.p"
    with config_path.open("wb") as config_file:
        pickle.dump([dataset_name, problem_parameters, ["x0", "xg"]], config_file)
    return config_path


def generate(dataset_name, num_train, num_test, seed=None, max_attempts=None):
    if seed is not None:
        np.random.seed(seed)

    problem_parameters = default_problem_parameters()
    config_path = write_config(dataset_name, problem_parameters)
    _, _, _, _, _, x_min, x_max, *_ = problem_parameters
    n, horizon, m = 4, problem_parameters[0], 3
    num_problems = num_train + num_test
    max_attempts = max_attempts or 10 * num_problems
    output_dir = SYSTEM_DIR / "data" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = Cartpole(config=str(config_path))
    params = {"x0": np.zeros((num_problems, n)), "xg": np.zeros((num_problems, n))}
    states = np.zeros((num_problems, n, horizon))
    controls = np.zeros((num_problems, m, horizon - 1))
    strategies = np.zeros((num_problems, 4, horizon - 1), dtype=int)
    costs = np.zeros(num_problems)
    solve_times = np.zeros(num_problems)

    index, attempts = 0, 0
    while index < num_problems and attempts < max_attempts:
        attempts += 1
        instance = {
            "x0": 0.5 * (x_min + x_max) + (np.random.rand(n) - 0.5) * (x_max - x_min),
            "xg": np.zeros(n),
        }
        try:
            success, cost, solve_time, solution = problem.solve_micp(instance, solver=cp.GUROBI)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            raise RuntimeError(f"Gurobi failed while solving sample {index}: {error}") from error

        if not success:
            print(f"Gurobi found no solution for sample {index}; resampling.")
            continue

        params["x0"][index] = instance["x0"]
        params["xg"][index] = instance["xg"]
        costs[index] = cost
        solve_times[index] = solve_time
        states[index], controls[index], strategies[index] = solution
        index += 1
        print(f"Solved {index}/{num_problems} (cost={cost:.4f}, time={solve_time:.4f}s)")

    if index != num_problems:
        raise RuntimeError(
            f"Generated {index}/{num_problems} feasible samples after {attempts} attempts. "
            "Increase --max-attempts or inspect the model/solver configuration."
        )

    train_data = [
        {key: value[:num_train] for key, value in params.items()},
        states[:num_train], controls[:num_train], strategies[:num_train],
        costs[:num_train], solve_times[:num_train],
    ]
    test_data = [
        {key: value[num_train:] for key, value in params.items()},
        states[num_train:], controls[num_train:], strategies[num_train:],
        costs[num_train:], solve_times[num_train:],
    ]
    for name, data in (("train.p", train_data), ("test.p", test_data)):
        with (output_dir / name).open("wb") as data_file:
            pickle.dump(data, data_file)
    print(f"Wrote {num_train} training and {num_test} test samples to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="default")
    parser.add_argument("--num-train", type=int, default=90)
    parser.add_argument("--num-test", type=int, default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--max-attempts", type=int,
        help="Maximum sampled MICP instances before failing (default: 10 per requested sample).",
    )
    args = parser.parse_args()
    if args.num_train < 0 or args.num_test < 0 or args.num_train + args.num_test == 0:
        parser.error("--num-train and --num-test must total at least one non-negative sample")
    generate(args.dataset_name, args.num_train, args.num_test, args.seed, args.max_attempts)


if __name__ == "__main__":
    main()
