# Actuator study contract, version 1

This contract is written before actuator-filter integration. It defines a new airborne
study model and does not change the accepted direct-wrench wind experiment.

## Coordinates and physical meanings

The body state is `x=[p_world(3), q_xyzw(4), v_world(3), omega_body(3)]`.
The unit scalar-last quaternion rotates body vectors into world coordinates; positive
body thrust is along body `+z`. The effort-lag state is
`y=[x(13), s(4)]`, explicitly distinguished from native 17-state RPM storage.

Motor indices are 0, 1, 2, 3 in the unchanged repository mixing order. The roll,
pitch and signed reaction-torque rows are respectively `[-1,-1,+1,+1]`,
`[-1,+1,+1,-1]`, and `[-1,+1,-1,+1]`. In the equivalent cross-product geometry,
`(r_x,r_y)` is `[(+L,-L),(-L,-L),(-L,+L),(+L,+L)]`. The implementation uses
`L` directly as the effective horizontal lever component, exactly as the accepted
mixing/native code does; it adds no factor of `sqrt(2)`. This clarifies the convention
rather than asserting that the parameter's historical “CoM-to-motor” label is accurate.
The last mixing row defines the reaction-torque signs; positive aerodynamic torque is
along body `+z`. No independent CW/CCW viewing convention is inferred from it.

The wrench order is `w=[collective_thrust_N, torque_body_x_Nm,
torque_body_y_Nm, torque_body_z_Nm]`. Its force map is
`B=[ones; L*mix[0]; L*mix[1]; thrust2torque*mix[2]]`.

* `u` is the commanded **nominal-equivalent per-motor effort**, in N, not RPM,
  actual force, wrench, voltage, PWM or normalized fraction.
* `s` is the internal nominal-equivalent effort state, also in N. It is not a
  second command and is not divided by effectiveness on plant input.
* Actual per-motor force is `f=eta*s` for A2 and `f=eta*u` for A1.
* Applied wrench is `B*f`. Effectiveness is applied exactly once and scales thrust
  and that motor's associated reaction torque together. Separate yaw-torque damage
  is a later variant. The surrogate omits rotor gyroscopic/inertial torque.

All comparisons retain the same nominal command bounds from `drones/params.toml`.
Partial effectiveness satisfies `0 < eta_i <= 1`. There is no command-limit expansion
to restore lost authority and no all-zero idle switch in this airborne model. An
initial actuator state may lie below the airborne minimum, but must lie in
`[0,u_max]`; bounded commands keep this interval invariant. Controllers/filters audit
bounds; the physical derivative/step never silently clips an invalid input.

## Two distinct models

A1 is a 13-state algebraic effectiveness-only model, `w=B diag(eta) u`. Its
functions never divide by a time constant. Identity effectiveness recovers the
accepted direct-wrench equations with the declared RK body integrator, not bitwise
identity to the legacy symplectic integrator.

A2 uses strictly positive finite time constants (seconds):
`ds/dt=(u-s)/tau`, `dx/dt=f_x(x)+G_x(x)B diag(eta)s`. Consequently
`f_a=[body_derivative(x,B diag(eta)s); -s/tau]`,
`G_a=[zeros(13,4); diag(1/tau)]`. A direct-wrench body input matrix cannot be used
as a command matrix for A2.

For a held command the exact motor solution is
`s(t+h)=s(t)+[-expm1(-h/tau)]*(u-s(t))`.
Each body RK4 substep evaluates this evolving effort at its start, half-time, and
end; final quaternions are normalized. End-of-hold force is never applied for the
whole hold. The analytic motor endpoint is exact up to arithmetic, whereas body
integration still has discretization error. A positive-lag limit requires refinement
of the body time step to resolve the initial actuator boundary layer.

An event changes the currently active model parameters and preserves all physical
body/motor state entries. Changing eta may instantly change force; changing tau
changes only derivatives. No motor state is reset to a new trim. The model provides
no future event schedule.

## Shared bounded allocation and metric

The input to every adapter is the same desired wrench from the declared body
controller, including any currently known wind/drag feedforward in that controller.
The adapter adds no second feedforward term. F0 ignores current effectiveness and
lag when constructing its command. F1 uses current effectiveness and ignores lag.
F2, shared unchanged by frozen/adaptive/DR/single-fallback methods, uses the current
motor observation and model to target the desired motor endpoint after one command
hold `Delta`: `s_des=(B^-1*w_des)/eta`,
`u_raw=s+(s_des-s)/[-expm1(-Delta/tau)]`.

Bounds are imposed by solving a normalized wrench least-squares problem. Define
`d=[1,L,L,thrust2torque]` and `H=diag(1/d)B`. The accepted mixing gives
`H.T H=4I`. Normalize each wrench residual by `d_i*sum(u_max-u_min)`.
The static F1 columns are proportional to columns of H; the F2 endpoint columns
are additionally scaled by `eta_i*(1-exp(-Delta/tau_i))`. They remain orthogonal.
Therefore independently clipping the unconstrained inverse to the real bounds is
**the exact global minimum of this declared coupled bounded wrench objective**.
Construction rejects mixing matrices that violate this identity; the formula is
not claimed to solve a differently weighted or generic nonorthogonal allocation.
This equal natural-scale objective does not impose collective-over-yaw priority.

Diagnostics distinguish raw and bounded command, actual endpoint wrench, its
residual against the request, normalized residual norm, saturation mask and
numerical validity. For F0/F1 applied endpoint diagnostics still use the actual
eta/tau/current motors. Infeasible requests are reported; clipping never certifies
exact recovery. A separate algebraic allocator reports algebraic delivered wrench.

The filter's positive-definite command metric is
`diag(1/(u_max-u_min)^2)`. Distances and command-polytope measures use these
normalized command coordinates, not wrench-box volumes. Allocation residual
normalization and filter command distance are separate named quantities.

## Time, observation and publication

The reference command period is 0.040 s and reference rollout duration is 1.2 s;
experiments record the actual configured values. Feedback is evaluated once per
command boundary. Integration/held-interval checks may subdivide the hold, but
cannot refresh commands there. Predictor and plant use the same declared hold.
A physical parameter event inside a hold splits integration at the event while
retaining the previous command and continuous actuator state.

Each control call freezes one current point-model snapshot and one immutable policy
snapshot. The primary oracle experiment observes current body/motor states and
current parameters, not future faults. Actual plant state, observed state, and
estimated/model parameters are separate fields even when numerically equal.
Teacher initialization uses the same full augmented state, with no hidden motor
reset. Learning retains optimizer state across faults, publishes every completed
finite update at the next permitted boundary, and records completion/publication/use
times and versions. These scheduling rules are enforced by the study runtime rather
than by the stateless physics functions.

## Native parameter provenance and validation boundaries

The nominal effort time constant is derived from local one-sided linearizations
of the native RPM equation at hover. For RPM `r0`, spin direction coefficients
`a,b` give `lambda=a+2*b*r0`; a local invertible thrust-coordinate change preserves
this pole. The symmetric surrogate chooses the arithmetic mean of the two poles,
`tau_nominal=2/(lambda_up+lambda_down)`. Directional values and operating-region
variation are retained in the parameter derivation artifact. This is an explicit
model reduction choice from repository coefficients, including historical TODOs,
not a new hardware measurement.

Hover feasibility solves all four wrench equations under bounds. Unique required
commands outside bounds prove infeasibility of that specified zero-rate level
hover for invertible B and positive eta; they do not prove all flight impossible.
Positive/negative torque headroom keeps collective and the other torque entries
fixed. Maneuver witnesses are separate numerical evidence; failure to find one is
`witness_not_found`, not `proven_infeasible`.

P0 is the matched differentiable surrogate. P1 independently codes continuous body
and effort dynamics/integration without calling its transition. P2 uses native RPM
states, asymmetric motor dynamics, separate thrust/torque curves and rotor inertial
effects; effort-to-RPM conversion selects the monotone physical thrust branch and
records residuals. P2 is model mismatch, not hardware validation. The aggregate
`so_rpy_rotor_drag` actuator is not an asymmetric per-motor replacement. Modeled
collision terminates airborne outcome evaluation; contact animation is separate.

For P2, motor observation maps actual RPM through the true native thrust curve before
effectiveness. It is therefore an exact motor-state/curve observation oracle, not an
implemented hardware sensor. Native rotor inertia is not scaled by effectiveness;
the aerodynamic thrust/reaction torque is. Native motor coefficients are multiplied by
`tau_nominal/tau_current` to apply the declared lag perturbation. On a thrust curve with
negative linear coefficient, effort zero maps to the upper monotone zero-thrust RPM
root, which is distinct from a stopped-rotor idle command. These conventions are tested
and do not equate the native plant to the effort surrogate.
