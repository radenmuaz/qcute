# Direct SSH to a TPU VM (bypassing `gcloud ... ssh` per-call overhead)

`gcloud compute tpus queued-resources ssh <qr-name>` works but re-validates TPU state and
re-preps the node on every single call (~3-25s overhead each time, more if there's a
maintenance/preemption event to detect). Once a TPU VM is up and you've SSH'd into it via
gcloud at least once (which propagates your `~/.ssh/google_compute_engine` public key to the
instance), you can talk to it directly with plain `ssh` and a multiplexed connection.

## 1. Get the actual node name and external IP

The queued-resource name (e.g. `tpu1r`) is not always the underlying node name — gcloud ssh's
own "Finished preparing node <name>." line reveals it (e.g. `tpu1r` queued resource -> node
`tpu1`). Then:

```bash
gcloud compute tpus tpu-vm describe <node-name> --project raden-tpu --zone <zone> \
  --format="yaml(networkEndpoints,state)"
```

gives `networkEndpoints[0].accessConfig.externalIp` and `state` (must be `READY`).

## 2. Plain direct SSH

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 \
  -i ~/.ssh/google_compute_engine muaz@<external_ip> "echo ok"
```

No OS Login is configured on this project (checked project/instance metadata — no
`enable-oslogin`), so this is plain metadata-based SSH key auth; the key gcloud already
propagated on the first `queued-resources ssh` call keeps working directly.

## 3. Persistent multiplexed connection (skip repeated handshakes)

```bash
mkdir -p ~/.ssh/controlmasters
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ControlMaster=auto -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p \
  -o ControlPersist=6h -i ~/.ssh/google_compute_engine -fN muaz@<external_ip>
```

Then every subsequent command reuses the open connection (~0.3s vs several seconds through
gcloud):

```bash
ssh -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p -i ~/.ssh/google_compute_engine \
  muaz@<external_ip> "<command>"
```

`scp` can use the same `-o ControlPath=...` flag to reuse the connection too.

## Caveats

- Direct ssh does **not** check TPU state first. If the TPU gets preempted (spot instances,
  see [TPU.md](../TPU.md)) mid-session, a direct `ssh`/multiplexed command just hangs until
  `ConnectTimeout`/TCP timeout instead of gcloud's immediate `PREEMPTED` state error. If a
  command unexpectedly hangs, check state with:
  `gcloud compute tpus queued-resources describe <qr-name> --project raden-tpu --zone <zone> --format="value(state.state)"`
- The multiplexed master survives across separate tool calls/shells as long as its process
  (`ControlPersist`) is alive — check with
  `ssh -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p -O check muaz@<external_ip>`.
- `pgrep -f`/`pkill -f` run over ssh will self-match the ssh command's own argument string if
  the pattern is a substring of the command you're running it from — use `ps aux | grep -i
  <name>` to eyeball real PIDs first when in doubt.
