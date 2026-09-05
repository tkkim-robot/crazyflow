# Separate motor-off contact continuations

These artifacts answer the request to show a failed flight collide or fall under contact dynamics. At an explicitly declared trigger, the recorded body position, orientation, world velocity and body angular velocity initialize a MuJoCo free body. Rotor thrust is then zero. Subsequent robot poses come from 1 ms `mj_step` integration, gravity, the recorded aerodynamic model and actual sphere/ground contact constraints. This is a motor-cut presentation policy, not the original controller's response. It does not change any archived controller result or prove controller safety after the handoff.

| Artifact | Handoff | First obstacle contact | First ground contact |
| --- | --- | --- | --- |
| `legacy-estimated-physical-contact-v2` | 4.885739 s, swept sphere contact | 4.886739 s | 5.574739 s |
| `static-impact-drop-fixture` | 0 s, synthetic motor-cut fixture | 0.176 s | 0.512 s |
| `seed209-adaptive-safety-abort-drop` | 13.840 s, first recorded degraded command | None | 14.596 s |

The seed 209 continuation is a safety-abort drop. Its trigger is not an observed obstacle collision. The legacy continuation uses the first swept crossing of the collision sphere, with quaternion interpolation and a backoff check against starting in penetration. Actual MuJoCo contacts are recorded independently after that handoff. The archived legacy encounter has stationary obstacles; the moving-surface regression below verifies the general moving-obstacle implementation separately.

## Body and collision model

The `cf21B_500` recorded point dynamics have mass 0.04338 kg and inertia `diag(25, 28, 49) × 10^-6 kg m²`. Navigation handoffs read the actual recorded mass/inertia at the trigger, including any payload already applied. These inertias differ from the visual XML's `diag(23.951, 23.951, 32.347) × 10^-6`. Both descriptions place the inertial center at the body origin; there is no center-of-mass offset to transform at handoff. The continuation preserves the recorded dynamics inertia.

The collider follows the existing XML: a sphere of radius 0.086 m centered at body offset `(0, 0, 0.02)` m. A sphere of radius 0.106 m around the body origin encloses it. Older navigation/competent benchmarks used a centered 0.05 m point envelope; their physical-clearance metrics remain metrics of that earlier envelope and must not be relabeled as XML-sphere or mesh contact. The rotor mesh is visual only. This contact model does not represent rotor breakage, flexible structures, or individual blade impacts.

MuJoCo free-joint quaternion storage is `wxyz`; recorded states use `xyzw`. The adapter preserves world linear velocity and body angular velocity. Gravity, body-frame linear aerodynamic drag, wind and world-frame external force/torque are applied to the free body. The direct-wrench implementation defines its external torque in world coordinates and transforms it into body coordinates for its angular equation. Navigation examples have zero external torque. Recorded body parameters are frozen at the trigger; these brief continuations do not simulate subsequent payload attachment events. Wind continues on the recorded absolute event schedule.

## Prescribed moving obstacles

Obstacle centers are evaluated on their original absolute-time trajectories. Navigation recordings use piecewise-linear interpolation, and the imposed velocity is the derivative of that same interpolant. The implementation refuses extrapolation outside recorded time support. The legacy scenario uses its archived constant-velocity formula.

Each obstacle is an externally driven free-joint sphere with a 1000 kg drive mass and gravity compensation. Its position and velocity are reset to the prescribed trajectory at each 1 ms boundary. This supplies the correct moving surface velocity to MuJoCo's contact Jacobian. A mocap body with positions alone would have zero solver velocity and would fail the immediate momentum-transfer regression. The finite drive mass approximates a prescribed environment; its small impact response is reset at the next boundary. The nominal drone/obstacle mass ratio is `4.338 × 10^-5`. This external drive can supply work; it is not an isolated momentum-conserving environment. Metadata reports maximum position and velocity corrections, which can also include changes at interpolation knots.

The identical-contact-pose test uses a stationary drone, negligible initial penetration and an obstacle velocity of either 0 or +0.5 m/s. After the first 1 ms step, the moving obstacle produces drone x velocity +0.158620 m/s; the stationary obstacle produces only +0.00000254 m/s. A longer moving-impact test confirms directional momentum transfer and exact prescribed obstacle positions. These checks distinguish velocity-aware contact from a position-only animation.

## Numerical limits and reproducibility

The model uses MuJoCo 3.12.0, `implicitfast`, a 1 ms step, a 6 ms soft-contact time constant, Newton contact solving, and explicit friction coefficients recorded in each XML/JSON. Contact softness permits temporary overlap: the maximum penetration in these three artifacts is approximately 8.27–9.77 mm. Contact force peaks depend on these numerical/material assumptions; they are not validated real-drone impact loads. Ground contact permits sliding or rolling under continuing wind and does not imply the body comes fully to rest.

Eight focused tests pass, covering inertia and collider geometry, exact initial-state transfer, ballistic fall, contact deflection and ground support, swept triggers, absolute obstacle time, moving-surface momentum, and refusal to invent an unsafe trigger. `CONTACT_ARTIFACT_AUDIT.json` records independently checked contact times and verifies the source, model, arrays and archived input hashes. Each artifact includes the exact command, a source snapshot, the executable contact XML, raw state/contact arrays and event metadata. Reproduction requires a fresh output directory. Run the focused tests with:

```sh
JAX_PLATFORMS=cpu .pixi/envs/gpu-tests/bin/python -m pytest tests/unit/safety/da_plcbf/test_contact_replay.py -q
.pixi/envs/gpu-tests/bin/python benchmark/da_plcbf_contact_artifact_audit.py --root artifacts/da_plcbf/hover-explanation-20260905/contact
```

The original `legacy-estimated-physical-contact` directory is retained unchanged with a `SUPERSEDED.md` notice: it used the preliminary mocap formulation. Use `legacy-estimated-physical-contact-v2` for the final demonstration. All preexisting controller artifacts and manifests remain intact.

MuJoCo's official descriptions of [free-joint state coordinates](https://mujoco.readthedocs.io/en/latest/overview.html#floating-objects) and [mocap bodies](https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-mocap) explain the coordinate transfer and why position-only mocap obstacles are insufficient for this contact-velocity contract.
