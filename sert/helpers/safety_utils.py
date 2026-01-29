from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from pyrep.backend import sim


@dataclass
class SafetyState:
    prev_distance: Optional[float] = None


def _safe_dt() -> float:
    try:
        return float(sim.simGetSimulationTimeStep())
    except Exception:
        return 0.05


def get_scene_safety_distance(env) -> float:
    try:
        d_now = float(getattr(env._task._scene, "_last_step_clearance"))
    except Exception:
        d_now = float("inf")
    if not np.isfinite(d_now):
        d_now = 1e3
    return d_now


def compute_safety_now(
    d_now: float,
    state: SafetyState,
    d_safe: float,
    ttc_max: float,
    dt: Optional[float] = None,
) -> Tuple[np.ndarray, SafetyState]:
    if dt is None:
        dt = _safe_dt()
    dt = max(1e-6, float(dt))

    rel_vel = 0.0
    if state.prev_distance is not None and np.isfinite(state.prev_distance) and np.isfinite(d_now):
        rel_vel = (d_now - state.prev_distance) / dt

    ttc = ttc_max
    if rel_vel < -1e-6 and np.isfinite(d_now):
        ttc = min(ttc_max, d_now / max(1e-6, -rel_vel))

    near_flag = 1.0 if d_now < d_safe else 0.0
    near_ratio = max(0.0, min(1.0, (d_safe - d_now) / max(1e-6, d_safe)))

    safety_now = np.array([d_now, ttc, near_ratio, near_flag, rel_vel], dtype=np.float32)
    state.prev_distance = d_now
    return safety_now, state


def compute_cost(
    safety_now: np.ndarray,
    d_safe: float,
    ttc_safe: float,
    rel_vel_safe: float,
    collision_distance: float,
    w_dist: float,
    w_ttc: float,
    w_near: float,
    w_rel_vel: float,
    w_collision: float,
) -> float:
    d_now, ttc, near_ratio, near_flag, rel_vel = safety_now.tolist()
    dist_term = max(0.0, (d_safe - d_now) / max(1e-6, d_safe))
    ttc_term = max(0.0, (ttc_safe - ttc) / max(1e-6, ttc_safe))
    rel_vel_term = max(0.0, (-rel_vel) / max(1e-6, rel_vel_safe))
    collision_flag = 1.0 if d_now < collision_distance else 0.0

    cost = (
        w_dist * dist_term
        + w_ttc * ttc_term
        + w_near * near_ratio
        + w_rel_vel * rel_vel_term
        + w_collision * collision_flag
    )
    return float(cost)
