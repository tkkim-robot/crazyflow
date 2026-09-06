# Review-only actuator study bundle

The sealed main campaign records 384/384 completed method episodes. Protocol SHA-256: `402b466f9fcf294e4c828cf391d04a72fe7b5e0bb6d0e54dfb2e3e2daab7631d`.

This compact archive includes 12787 original files; 8057 inventoried files remain local-only. The detached manifest.json records the original repository-relative path, exact SHA-256, byte size, and inclusion or omission reason for every source file. Failed, interrupted, negative, and tuning metadata use the same inclusion rule as successful metadata. Collection does not rerun or validate the scientific results.

This is an incomplete subset for full physical replay. Most rollout/control arrays, event streams, and saved learner snapshots are omitted. The selected candidate F2/A dense and command-application arrays do not include the controls needed by the full video replay validator. Shared-state probe arrays and selected initial deployments are included; their broader dependencies may remain local. Video files and optimized compilation-cache binaries are local-only. The selected PNG is an audit poster, not a new experiment. Exact float32 replay additionally depends on compatible preserved optimized executables and the recorded software/GPU environment.

Original members preserve repository-relative paths and exact bytes. Generated review/checksum members are under __publication__/. Tar entries have mtime 0, uid/gid 0, empty owner names, and regular-file mode 0644. Source files are never overwritten.

Verify every included byte, without extraction or Git:

```sh
python -m benchmark.da_plcbf_actuator_publication verify --publication-dir /absolute/path/to/publication-directory
```

The verifier checks the detached archive digest, exact member set, member sizes and digests, canonical headers and SHA256SUMS. It rejects unexpected members, traversal, duplicate names, symlinks and other nonregular entries. Authenticity of the whole bundle still requires retaining the detached manifest digest through a trusted channel.
