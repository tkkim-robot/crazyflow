"""Command governor retaining a checked, phase-consistent actuator backup.

The PL-CBF remains the proposed command generator. A proposal must also leave a
checked continuation over the remaining observed prediction window. Completed
learner updates remain unrestricted; an executing backup owns its immutable
parameter snapshot, anchor, and maneuver clock independently of publications.

These are numerical finite-horizon checks, not a terminal-invariant-set proof or
a guarantee against unobserved future dynamics changes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.actuator_dynamics import effort_lag_step
from crazyflow.safety.da_plcbf.actuator_learning import (
    acceleration_to_actuator_command,
    actuator_skill_actions,
)
from crazyflow.safety.da_plcbf.actuator_plcbf import (
    ActuatorRollouts,
    check_actuator_hold,
    operational_node_margins,
)
from crazyflow.safety.da_plcbf.continuous_version_a import runtime_policy_values


def context_rollouts(
    params: Any, spec: Any, initial: Any, model: Any, actor: Any, anchor: Any, elapsed_seconds: Any
) -> ActuatorRollouts:
    """Continue a snapshot's feedback policies without restarting anchor or phase."""
    count = spec.latent_codes.shape[0]
    current = jnp.broadcast_to(initial, (count, 17))
    period = actor.dt * actor.control_interval_steps

    def hold(state: Any, boundary: Any) -> tuple[Any, Any]:
        phase = (elapsed_seconds + boundary * period) / (actor.dt * actor.horizon)
        acceleration = actuator_skill_actions(params, spec, state, anchor, phase, actor)
        allocation = acceleration_to_actuator_command(acceleration, state, model, actor)

        def integrate(value: Any, _: Any) -> tuple[Any, Any]:
            following = effort_lag_step(
                value, allocation.command, model, actor.dt, substeps=actor.plant_substeps
            )
            return following, following

        final, future = jax.lax.scan(
            integrate,
            state,
            None,
            length=actor.control_interval_steps,
            unroll=min(actor.control_interval_steps, 2),
        )
        commands = jnp.broadcast_to(allocation.command, (actor.control_interval_steps, count, 4))
        return final, (future, commands)

    _, (future, commands) = jax.lax.scan(
        hold,
        current,
        jnp.arange(
            (actor.horizon + actor.control_interval_steps - 1) // actor.control_interval_steps
        ),
        unroll=actor.rollout_scan_unroll,
    )
    future = jnp.swapaxes(future.reshape((-1, count, 17))[: actor.horizon], 0, 1)
    commands = jnp.swapaxes(commands.reshape((-1, count, 4))[: actor.horizon], 0, 1)
    states = jnp.concatenate((current[:, None], future), axis=1)
    valid = jnp.all(jnp.isfinite(states), axis=(1, 2)) & jnp.all(
        jnp.isfinite(commands), axis=(1, 2)
    )
    return ActuatorRollouts(states, commands, valid)


def checked_paths(batch: Any, model: Any, obstacles: Any, safety: Any, config: Any) -> tuple:
    """Check every path's collision, operational, command and effort constraints."""
    collision = runtime_policy_values(
        batch.states[..., :13],
        obstacles,
        obstacle_clearance=config.obstacle_clearance,
        ego_radius=config.ego_radius,
    )
    operational = jax.vmap(
        lambda nodes: operational_node_margins(
            nodes, safety, arena_clearance=config.arena_clearance
        )
    )(batch.states)
    operational_min = jnp.min(operational, axis=(1, 2))
    command_min = jnp.minimum(
        jnp.min(batch.commands - model.command_lower, axis=(1, 2)),
        jnp.min(model.command_upper - batch.commands, axis=(1, 2)),
    )
    effort_min = jnp.minimum(
        jnp.min(batch.states[..., 13:], axis=(1, 2)),
        jnp.min(model.command_upper - batch.states[..., 13:], axis=(1, 2)),
    )
    valid = (
        batch.valid
        & collision.input_valid
        & jnp.isfinite(operational_min)
        & (collision.values >= -config.tolerance)
        & (operational_min >= -config.operational_tolerance)
        & (command_min >= -config.tolerance)
        & (effort_min >= -config.tolerance)
    )
    return valid, collision.values, operational_min


@dataclass(frozen=True)
class BackupSnapshot:
    params: Any
    anchor: Any
    started_at: float
    skill_index: int
    generation: int
    certified_until: float


class CommittedBackupController:
    """Episode-owned governor; warmup uses commit=False and cannot retain state."""

    def __init__(self, functions: Any, spec: Any, actor: Any, config: Any) -> None:
        if config.horizon <= config.command_hold_steps:
            raise ValueError("backup checking needs a prediction window beyond the command hold")
        self.base = functions
        self.nominal, self.emergency, self.candidates = (
            functions.nominal,
            functions.emergency,
            functions.candidates,
        )
        self.controller = functions.controller
        self.spec, self.actor, self.config = spec, actor, config
        self.backup: BackupSnapshot | None = None
        self.generation = 0
        self._held = jax.jit(
            lambda state, command, model, obstacles, safety: check_actuator_hold(
                state, command, model, obstacles, safety, config
            )
        )
        self._checked = jax.jit(
            lambda batch, model, obstacles, safety: checked_paths(
                batch, model, obstacles, safety, config
            )
        )

        def evaluate(
            state: Any,
            params: Any,
            anchor: Any,
            elapsed: Any,
            model: Any,
            obstacles: Any,
            safety: Any,
            *,
            successor: bool,
            prefix_steps: Any = None,
        ) -> tuple:
            batch = context_rollouts(params, spec, state, model, actor, anchor, elapsed)
            if successor:
                count = config.command_hold_steps
                batch = ActuatorRollouts(
                    batch.states[:, :-count], batch.commands[:, :-count], batch.valid
                )
                obstacles = obstacles._replace(
                    centers=obstacles.centers[count:],
                    mask=obstacles.mask[count:],
                    velocities=None
                    if obstacles.velocities is None
                    else obstacles.velocities[count:]
                    if obstacles.velocities.ndim == 3
                    else obstacles.velocities,
                )
            if prefix_steps is not None:
                # Preserve exactly the previously checked interval. Unseen extension
                # nodes cannot invalidate an otherwise executable stored tail.
                end = jnp.clip(prefix_steps, 1, config.horizon)
                states = batch.states[:, jnp.minimum(jnp.arange(config.horizon + 1), end)]
                commands = batch.commands[:, jnp.minimum(jnp.arange(config.horizon), end - 1)]
                batch = ActuatorRollouts(
                    states,
                    commands,
                    jnp.all(jnp.isfinite(states), axis=(1, 2))
                    & jnp.all(jnp.isfinite(commands), axis=(1, 2))
                    & (prefix_steps >= config.command_hold_steps),
                )
                obstacles = obstacles._replace(
                    mask=obstacles.mask & (jnp.arange(config.horizon + 1) <= end)[:, None]
                )
            valid, collision, operational = checked_paths(batch, model, obstacles, safety, config)
            return valid, collision, operational, batch.commands[:, 0], batch.states

        self._current = jax.jit(lambda *args: evaluate(*args, successor=False))
        self._successor = jax.jit(lambda *args: evaluate(*args, successor=True))
        self._remaining = jax.jit(
            lambda *args: evaluate(*args[:-1], successor=False, prefix_steps=args[-1])
        )

    def controller_at(
        self,
        state: Any,
        params: Any,
        model: Any,
        obstacles: Any,
        safety: Any,
        previous: Any,
        goal: Any,
        *,
        when: float,
        commit: bool,
    ) -> Any:
        from crazyflow.safety.da_plcbf.actuator_experiment import _hash_tree

        base = self.base.controller(state, params, model, obstacles, safety, previous, goal)
        if base.certificates.rollouts.states.shape[0] != self.spec.latent_codes.shape[0] + 1:
            raise ValueError("committed backup requires one mission and one matching skill bank")
        period = self.config.command_period
        retained = self.backup if commit else None
        selected_backup = retained
        generation = self.generation
        proposed = np.asarray(base.action)
        proposal_check = jax.block_until_ready(
            self._held(state, proposed, model, obstacles, safety)
        )
        successor = proposal_check.nodes[-1]
        proof_valid = False
        successor_kind = "none"
        successor_margin = float("nan")
        full_until = when + self.config.horizon * self.config.dt
        proof_states = np.full((self.config.horizon + 1, 17), np.nan, dtype=np.float32)
        proof_nodes = 0
        used_stored_tail = False
        retained_age = when - retained.started_at if retained is not None else 0.0
        if retained is not None and bool(proposal_check.passed):
            assert retained_age >= -1e-9, "a stored backup cannot begin after the sensing boundary"
            values = jax.block_until_ready(
                self._successor(
                    successor,
                    retained.params,
                    retained.anchor,
                    jnp.asarray(max(0.0, retained_age) + period, state.dtype),
                    model,
                    obstacles,
                    safety,
                )
            )
            proof_valid = bool(values[0][retained.skill_index])
            successor_margin = float(values[1][retained.skill_index])
            successor_kind = "retained" if proof_valid else "none"
            if proof_valid:
                selected_backup = replace(retained, certified_until=full_until)
        if not proof_valid and bool(proposal_check.passed):
            values = jax.block_until_ready(
                self._successor(
                    successor,
                    params,
                    successor[:3],
                    jnp.asarray(0.0, state.dtype),
                    model,
                    obstacles,
                    safety,
                )
            )
            valid, scores = np.asarray(values[0]), np.asarray(values[1])
            if np.any(valid):
                target = int(np.argmax(np.where(valid, scores, -np.inf)))
                generation += 1
                selected_backup = BackupSnapshot(
                    params, successor[:3], when + period, target, generation, full_until
                )
                proof_valid, successor_kind = True, "fresh"
                successor_margin = float(scores[target])
        accepted_proposal = bool(proposal_check.passed) and proof_valid
        if accepted_proposal:
            hold_nodes = np.asarray(proposal_check.nodes)[
                : self.config.command_hold_steps
                * self.config.held_substeps : self.config.held_substeps
            ]
            proof_states = np.concatenate(
                (hold_nodes, np.asarray(values[4][selected_backup.skill_index])), axis=0
            )
            proof_nodes = len(proof_states)
        executed_backup = False
        action, applied = proposed, proposal_check
        backup_values = None
        if not accepted_proposal:
            if retained is not None:
                backup_values = jax.block_until_ready(
                    self._current(
                        state,
                        retained.params,
                        retained.anchor,
                        jnp.asarray(max(0.0, retained_age), state.dtype),
                        model,
                        obstacles,
                        safety,
                    )
                )
                if bool(backup_values[0][retained.skill_index]):
                    action = np.asarray(backup_values[3][retained.skill_index])
                    selected_backup = replace(retained, certified_until=full_until)
                    executed_backup = True
                elif retained.certified_until >= when + period - 1e-9:
                    remaining = int(
                        np.floor((retained.certified_until - when) / self.config.dt + 1e-8)
                    )
                    backup_values = jax.block_until_ready(
                        self._remaining(
                            state,
                            retained.params,
                            retained.anchor,
                            jnp.asarray(max(0.0, retained_age), state.dtype),
                            model,
                            obstacles,
                            safety,
                            jnp.asarray(remaining, jnp.int32),
                        )
                    )
                    if bool(backup_values[0][retained.skill_index]):
                        action = np.asarray(backup_values[3][retained.skill_index])
                        selected_backup = retained
                        executed_backup = True
                        used_stored_tail = True
                if executed_backup:
                    proof_states = np.asarray(backup_values[4][retained.skill_index])
                    proof_nodes = (
                        min(
                            self.config.horizon,
                            int(round((selected_backup.certified_until - when) / self.config.dt)),
                        )
                        + 1
                    )
            if not executed_backup:
                batch = base.certificates.rollouts
                # Mission candidate zero is not a reusable fallback policy.
                fresh = ActuatorRollouts(batch.states[1:], batch.commands[1:], batch.valid[1:])
                valid, scores, _ = jax.block_until_ready(
                    self._checked(fresh, model, obstacles, safety)
                )
                valid, scores = np.asarray(valid), np.asarray(scores)
                if np.any(valid):
                    target = int(np.argmax(np.where(valid, scores, -np.inf)))
                    generation += 1
                    selected_backup = BackupSnapshot(
                        params, state[:3], when, target, generation, full_until
                    )
                    action = np.asarray(fresh.commands[target, 0])
                    executed_backup = True
                    proof_states = np.asarray(fresh.states[target])
                    proof_nodes = len(proof_states)
            applied = jax.block_until_ready(self._held(state, action, model, obstacles, safety))
            if not bool(applied.passed):
                # Never label a failed immediate physical check as a checked backup.
                executed_backup = False
                selected_backup = None
                action = np.asarray(base.action)
                applied = proposal_check
                proof_nodes = 0
        if commit:
            self.backup = selected_backup
            self.generation = generation
        backup_audit = {
            "proposal_accepted": accepted_proposal,
            "proposal_original_action": proposed,
            "proposal_original_qp_valid": bool(base.qp_valid),
            "successor_backup_kind": successor_kind,
            "successor_collision_margin": successor_margin,
            "executed_backup": executed_backup,
            "used_stored_tail": used_stored_tail and executed_backup,
            "has_checked_plan": accepted_proposal or executed_backup,
            "remaining_checked_seconds": selected_backup.certified_until - when
            if (accepted_proposal or executed_backup) and selected_backup is not None
            else 0.0,
            "checked_trajectory": proof_states,
            "checked_trajectory_nodes": proof_nodes,
            "backup_skill_index": selected_backup.skill_index if selected_backup else -1,
            "backup_generation": selected_backup.generation if selected_backup else -1,
            "backup_anchor": np.asarray(selected_backup.anchor)
            if selected_backup
            else np.full(3, np.nan),
            "backup_started_at": selected_backup.started_at if selected_backup else float("nan"),
            "backup_certified_until": selected_backup.certified_until
            if selected_backup
            else float("nan"),
            "backup_params_sha256": _hash_tree(selected_backup.params) if selected_backup else "",
        }
        values = base._asdict()
        if not accepted_proposal:
            values.update(
                action=action,
                applied=applied,
                next_estimated_state=applied.nodes[-1],
                qp_valid=False,
                fallback_valid=False,
                executed_policy_dual=jnp.asarray(0.0),
                execution_mode=jnp.asarray(5 if executed_backup else 3),
                degraded=jnp.asarray(not executed_backup),
                intervention_norm=jnp.linalg.norm(
                    (action - base.nominal_action) / (model.command_upper - model.command_lower)
                ),
                policy_residual=base.certificates.bounds[int(base.selected_index)]
                - base.certificates.rows[int(base.selected_index)] @ action,
            )
        return SimpleNamespace(**values, backup_audit=backup_audit)
