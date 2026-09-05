"""Capture compiler-variant rollout, objective gradient, and persistent-update numerics."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.persistent_skill_learner import rollout_skill_library
from crazyflow.safety.da_plcbf.state_conditioned_learning import build_reference_skill_learner_from_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
checkpoint = Path('artifacts/da_plcbf/case-study-20260905/atlas-v1/uncompensated/t0100')
bundle, contract, learner = build_reference_skill_learner_from_checkpoint(checkpoint)
states = jnp.concatenate((bundle.physical_state[None], contract.anchors[jnp.array((0, 3, 11))]))
rollout = jax.jit(jax.vmap(lambda state, model: rollout_skill_library(bundle.state.params, bundle.spec, state, model, bundle.actuator, bundle.config), in_axes=(0, None)))
grad = jax.jit(jax.value_and_grad(learner.loss, has_aux=True))
groups = {
 'wind_rollouts': rollout(states, bundle.point_model),
 'nominal_rollouts': rollout(states, contract.model),
 'objective_and_gradient': grad(bundle.state.params, bundle.physical_state, bundle.point_model, bundle.state.previous_params, bundle.state.library_version),
 'persistent_update': learner.step(bundle.state, bundle.physical_state, bundle.point_model),
}
jax.block_until_ready(groups)
arrays = {}
for name, tree in groups.items():
 for index, leaf in enumerate(jax.tree.leaves(tree)):
  arrays[f'{name}_{index:03d}'] = np.asarray(leaf)
np.savez_compressed(args.output, **arrays)
args.output.with_suffix('.json').write_text(json.dumps({
 'checkpoint_npz_sha256': bundle.sha256,
 'quad_rollouts_source_sha256': hashlib.sha256(Path('crazyflow/safety/da_plcbf/quad_rollouts.py').read_bytes()).hexdigest(),
 'tolerance_predeclared': {'floating_point_rtol': 0.00003, 'floating_point_atol': 0.0000001, 'booleans_and_integers': 'exact'},
 'groups': {name: str(jax.tree.structure(tree)) for name, tree in groups.items()},
 'arrays': {name: {'shape': list(array.shape), 'dtype': array.dtype.str} for name,array in arrays.items()},
}, indent=2)+'\n')
print(args.output, flush=True)
