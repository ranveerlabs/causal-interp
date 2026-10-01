# xp on the server

The FreeBSD laptop needs SSH and a terminal. CUDA, Python, model weights and all
compute stay on the Linux server. Allow several GB for dependencies and model
cache, plus result storage. Use a server account with Python 3.10–3.12, git,
tmux and working NVIDIA drivers. No root or driver changes are needed.

```sh
ssh YOUR_USER@LambdaV
git clone https://github.com/ranveerlabs/causal-interp.git
cd causal-interp
./xp/setup.sh
tmux new -s xp './xp/run.sh'
```

Git needs your existing author/committer configuration on the server for local
checkpoints. Check `git var GIT_AUTHOR_IDENT` before leaving. If unset, configure
your own name and email in this clone. Do not copy an invented identity.

Setup creates `xp/.venv`, pins torch 2.7.1+cu126 and installs the remaining project
requirements with that torch constraint. It prints any hardware mismatch, checks
sm_60 and avoids occupied GPUs. It leaves the project's existing `.venv` alone.
A dependency conflict or missing compatible GPU stops setup. It does not change
system CUDA or NVIDIA drivers.

Preflight shows git and plan state, hardware, compute processes, queue counts,
output location and rough time/disk estimates. It uses a small synthetic probe,
then reproduces the frozen Phase 1 baseline. A failed reproduction stops all
experiments. There is a ten-second countdown after successful preflight.

```sh
./xp/run.sh --hours 8 --gpus 0,1 --yes
```

`--hours` includes preflight time. At its deadline, safe in-flight units finish,
so exit can be later. `--gpus` selects physical nvidia-smi indices. Occupied or
incompatible cards are excluded. `--yes` skips the countdown. No arguments are
required. Every worker uses one GPU and fp32. Foreign compute causes xp to yield
its own worker. If all GPUs are released, pending work waits for your next run.

Detach with Ctrl-b, then d. Disconnect SSH normally. Reconnect and inspect:

```sh
ssh YOUR_USER@LambdaV
cd causal-interp
./xp/status.sh
cat xp/DIGEST.md
tmux attach -t xp
```

Status is read-only and shows heartbeat age, running/stopped state, units, GPU
states, active work, errors, ETA, latest completion and recent logs. A stale
heartbeat after a hard kill remains visible. Results are in `xp/results/`, logs
in `xp/logs/gpuN.log`, `xp/logs/cpu.log` and `xp/logs/gate.log`. The heartbeat is
also available as `xp/logs/heartbeat.txt`.

```sh
touch xp/STOP
./xp/status.sh
```

STOP stops new claims and lets safe units finish. SIGINT and SIGTERM behave the
same way. The STOP file remains until you remove it:

```sh
rm xp/STOP
./xp/run.sh
```

Resume validates final checksums, skips valid outputs, recovers dead-worker claims
and recomputes incomplete/corrupt outputs. Leave the same PLAN and executable
files in place. A changed fingerprint refuses reuse. Do not delete the SQLite
queue or edit payloads to force a result. One launcher per clone is enforced.
A local filesystem with working flock, SQLite locks and atomic rename is required.
Use a local disk on LambdaV, not an NFS home directory for the clone.

Each experiment gets a local git checkpoint only after all its core units are
DONE, FAILED or BLOCKED and the digest is regenerated. Existing staged changes
defer the checkpoint. No server code pushes anything. When back, review:

```sh
git log --oneline
git status --short
git diff origin/main..HEAD -- xp
```

Transfer/fetch the server's local commits to your development machine, review
`xp/DIGEST.md`, the individual payloads and failures, then push from that machine.
Do not push from the university server.

## Development checks

`python xp/smoke.py` uses temporary directories, synthetic work and a tiny random
transformer when torch/TransformerLens are installed. It never runs registered
experiments or downloads pretrained weights. Numerical checks need scipy.
For the small CUDA worker checks, set `XP_SMOKE_GPU` to a free GPU UUID from
nvidia-smi. They run only a 32-float fp32 kernel using the CUDA driver. They refuse
an existing foreign compute context. GPU contention and reproduction failure
are simulated, clearly separated from actual model execution in the development
report. Neither RTX results nor CPU checks certify P100 compatibility.
