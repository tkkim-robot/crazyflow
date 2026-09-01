# DA-PLCBF paired scientific report

Scheduled outcomes: 8; complete: 8; execution failures: 0.

Execution completeness, final-claim eligibility, and metric-level superiority are separate gates.

- Execution complete: `true`
- Final-claim eligible: `false`
- Global confirmatory superiority across every predeclared condition, baseline, and endpoint: `false`
- Supported confirmatory comparisons: `0` of `12`
- Exploratory comparisons (never claim-eligible): `24`

## Predeclared analysis roles

Confirmatory endpoints are `operational_failure`, trial-level `any_failure`, and `minimum_hard_margin`. Their Bonferroni family spans every condition and full-method-versus-baseline pairing.

For a declared safety-controller method, `operational_failure` includes execution failure, physical trace failure, or any explicit degraded interval. `nominal_only` has no safety controller, so its intentional certificate-unavailable marker is excluded while physical failures remain counted.

The confirmatory percentile bootstrap uses `50000` replicates across `12` comparisons, yielding `104.167` expected draws beyond each adjusted endpoint.

Certification coverage, degraded duration, intervention, controller latency, command-ready latency, and wall-step latency are exploratory diagnostics with unadjusted intervals; they cannot support a superiority statement.

## Claim blockers

- 4 completed runs failed method claim gates
- methods are not exactly the seven ordered core method IDs
- fewer than 100 paired trials per condition
- schedule was not predeclared for a final claim
- schedule is not a predeclared >=100-pair final-claim schedule

## Paired inference

| Role | Condition | Baseline | Metric | Pairs | Missing | Superiority | Conclusion |
|---|---|---|---|---:|---:|---|---|
| confirmatory | static | nominal_only | operational_failure | 1 | 0 | false | not eligible for a final claim: schedule must be predeclared with at least 100 matched trials |
| confirmatory | static | nominal_only | any_failure | 1 | 0 | false | not eligible for a final claim: schedule must be predeclared with at least 100 matched trials |
| confirmatory | static | nominal_only | minimum_hard_margin | 1 | 0 | false | not eligible for a final claim: schedule must be predeclared with at least 100 matched trials |
| confirmatory | dynamics_change | nominal_only | operational_failure | 1 | 0 | false | not eligible for a final claim: schedule must be predeclared with at least 100 matched trials |
| confirmatory | dynamics_change | nominal_only | any_failure | 1 | 0 | false | not eligible for a final claim: schedule must be predeclared with at least 100 matched trials |
| confirmatory | dynamics_change | nominal_only | minimum_hard_margin | 1 | 0 | false | not eligible for a final claim: schedule must be predeclared with at least 100 matched trials |
| confirmatory | ballistic_ball | nominal_only | operational_failure | 1 | 0 | false | not eligible for a final claim: schedule must be predeclared with at least 100 matched trials |
| confirmatory | ballistic_ball | nominal_only | any_failure | 1 | 0 | false | not eligible for a final claim: schedule must be predeclared with at least 100 matched trials |
| confirmatory | ballistic_ball | nominal_only | minimum_hard_margin | 1 | 0 | false | not eligible for a final claim: schedule must be predeclared with at least 100 matched trials |
| confirmatory | interceptor_drone | nominal_only | operational_failure | 1 | 0 | false | not eligible for a final claim: schedule must be predeclared with at least 100 matched trials |
| confirmatory | interceptor_drone | nominal_only | any_failure | 1 | 0 | false | not eligible for a final claim: schedule must be predeclared with at least 100 matched trials |
| confirmatory | interceptor_drone | nominal_only | minimum_hard_margin | 1 | 0 | false | not eligible for a final claim: schedule must be predeclared with at least 100 matched trials |
| exploratory | static | nominal_only | certified_time_fraction | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | static | nominal_only | degraded_duration_seconds | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | static | nominal_only | intervention_integral | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | static | nominal_only | controller_p99_seconds | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | static | nominal_only | command_ready_p99_seconds | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | static | nominal_only | wall_step_p99_seconds | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | dynamics_change | nominal_only | certified_time_fraction | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | dynamics_change | nominal_only | degraded_duration_seconds | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | dynamics_change | nominal_only | intervention_integral | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | dynamics_change | nominal_only | controller_p99_seconds | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | dynamics_change | nominal_only | command_ready_p99_seconds | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | dynamics_change | nominal_only | wall_step_p99_seconds | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | ballistic_ball | nominal_only | certified_time_fraction | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | ballistic_ball | nominal_only | degraded_duration_seconds | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | ballistic_ball | nominal_only | intervention_integral | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | ballistic_ball | nominal_only | controller_p99_seconds | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | ballistic_ball | nominal_only | command_ready_p99_seconds | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | ballistic_ball | nominal_only | wall_step_p99_seconds | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | interceptor_drone | nominal_only | certified_time_fraction | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | interceptor_drone | nominal_only | degraded_duration_seconds | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | interceptor_drone | nominal_only | intervention_integral | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | interceptor_drone | nominal_only | controller_p99_seconds | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | interceptor_drone | nominal_only | command_ready_p99_seconds | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |
| exploratory | interceptor_drone | nominal_only | wall_step_p99_seconds | 1 | 0 | false | exploratory only: this endpoint was not predeclared for confirmatory superiority |

> Conclusions apply only to the predeclared finite simulation horizon and recorded matched scenario tapes; they are not hardware or real-world safety guarantees.
