# Diagnostic review archive

Exact source metadata, protocols and amendments, initial checkpoints, evidence tables and figures are included. Large raw rollouts/event streams, learner arrays and movies remain local and are explicitly hashed in inventory.json. This is a review archive, not a self-contained full numerical replay bundle. All original protocol trial IDs must have a completed physical attempt before collection. Outcomes and validity are separate: completed failures remain failures.

Archive members retain repository-relative paths. Verify without extraction using `python -m benchmark.da_plcbf_actuator_diagnostic_evidence verify --output <this-directory>`. `--verify-local` additionally checks the current raw files against the complete inventory.
