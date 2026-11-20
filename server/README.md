# tensorquill-server

The server and offline detector binary for TensorQuill (the `tensorquill` package),
a debugger for PyTorch training runs. It takes a run's readings over Arrow Flight and writes
partitioned Parquet to local disk, S3 or GCS, and its `detect` subcommand reads a recorded run and
reports what went wrong in it. The `tensorquill` Python package runs that subcommand for you.

```
pip install tensorquill-server
tensorquill-server --help
```

This package holds no Python code. On the platforms below the wheel carries one program,
`tensorquill-server`, which pip puts in the environment's scripts directory (`bin/` in a virtual
environment):

| Platform | Wheel |
|---|---|
| Linux x86_64, glibc 2.17 or newer | `manylinux_2_17_x86_64` |
| Linux arm64, glibc 2.17 or newer | `manylinux_2_17_aarch64` |
| macOS arm64 | `macosx_11_0_arm64` |

On every other platform (Windows, Intel macOS, Alpine and other musl Linux) pip installs a
fallback wheel that holds the license files and nothing else: the install succeeds and there is
no `tensorquill-server` command. The `tensorquill` package then uses its Python detectors for
one-shot reads. To get the program there, build it from the source with `cargo build --release`
in the `server` directory.

The `tensorquill` package and this program must speak the same protocol number
(`tensorquill-server protocol` prints it; 3 since Nov 19, 2026). The package refuses a binary
with another number for offline detection and says which to install; streaming still works
across the two.

Beside the Parquet parts, each rank's folder in the store holds `incidents.json` (what the
detectors found) and `attempts.json` (the order in which the job's attempts were first seen,
which decides which attempt owns the rank after a restart). The run's `watchdog.json` says
whether each rank is making progress; a rank whose capture stopped while training went on
reads `capture_stopped`, which is neither a hang nor finished.

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option.
