"""Widen a trained checkpoint for the speed-commanded observation.

Turning on ``obs.command_conditioning`` appends ``object_vel``(6) +
``speed_command``(1) + ``dist_to_subtarget``(1) to the actor obs, so the LSTM's
input grows by 8.  rl_games refuses to load a checkpoint whose shapes differ,
so the old weights have to be *widened* rather than just renamed.

The new input columns are initialised to **zero**, which means right after
loading the network behaves exactly like the old policy (the extra inputs
contribute nothing).  Training then teaches it to use them — no behaviour
cliff, which is the whole point of a warm start.

    python isaacsimenvs/utils/widen_checkpoint.py \
        --input  outputs/.../last/model.pth \
        --output outputs/.../last/model_widened.pth \
        --add 8

Running-mean/std needs the new channels too: pass per-channel values with
``--new_mean``/``--new_std`` (defaults are sized for [object_vel(6),
speed_command, dist_to_subtarget] in normalised units).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

DEFAULT_MEAN = [0.0] * 6 + [0.35, 0.1]      # lin/ang vel ~0 ; cmd/v_ref ; dist (m)
DEFAULT_STD = [0.3] * 6 + [0.3, 0.1]


def _widen_2d(tensor: torch.Tensor, add: int, where: str = "col") -> torch.Tensor:
    """Append zero columns (or rows) so the shape grows by `add`."""
    if add <= 0:
        return tensor
    if where == "col":
        pad = torch.zeros(tensor.shape[0], add, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad], dim=1)
    pad = torch.zeros(add, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=0)


def _widen_1d(tensor: torch.Tensor, add: int, values: list[float]) -> torch.Tensor:
    if add <= 0:
        return tensor
    new = torch.tensor(values[:add], dtype=tensor.dtype, device=tensor.device)
    if new.numel() < add:  # pad if the caller gave fewer values
        new = torch.cat([new, torch.zeros(add - new.numel(), dtype=tensor.dtype)])
    return torch.cat([tensor, new])


def widen(ckpt: dict, add: int, new_mean: list[float], new_std: list[float]) -> dict:
    out = {}
    for policy_idx, payload in ckpt.items():
        if not isinstance(payload, dict):
            out[policy_idx] = payload
            continue
        new_payload = dict(payload)

        model = payload.get("model")
        if isinstance(model, dict):
            new_model = {}
            for name, tensor in model.items():
                if not torch.is_tensor(tensor):
                    new_model[name] = tensor
                    continue
                if name.endswith("rnn.weight_ih_l0"):
                    # LSTM input projection: (4*hidden, input) -> +add columns.
                    new_model[name] = _widen_2d(tensor, add, where="col")
                elif name.startswith("running_mean_std.") and name.endswith("running_mean"):
                    # input normaliser — NOT value_mean_std (that one is 1-wide)
                    new_model[name] = _widen_1d(tensor, add, new_mean)
                elif name.startswith("running_mean_std.") and name.endswith("running_var"):
                    new_model[name] = _widen_1d(tensor, add, [v * v for v in new_std])
                else:
                    new_model[name] = tensor
            new_payload["model"] = new_model

        nets = payload.get("assymetric_vf_nets")
        if isinstance(nets, dict):
            new_nets = {}
            for net_key, net in nets.items():
                if isinstance(net, dict):
                    new_net = {}
                    for name, tensor in net.items():
                        if not torch.is_tensor(tensor):
                            new_net[name] = tensor
                            continue
                        if name.endswith("rnn.weight_ih_l0"):
                            new_net[name] = _widen_2d(tensor, add, where="col")
                        elif name.startswith("running_mean_std.") and name.endswith("running_mean"):
                            # critic sees command + dist only (object_vel is
                            # already in state_list) -> 2 new channels
                            new_net[name] = _widen_1d(tensor, 2, new_mean[-2:])
                        elif name.startswith("running_mean_std.") and name.endswith("running_var"):
                            new_net[name] = _widen_1d(tensor, 2, [v * v for v in new_std[-2:]])
                        else:
                            new_net[name] = tensor
                    new_nets[net_key] = new_net
                else:
                    new_nets[net_key] = net
            new_payload["assymetric_vf_nets"] = new_nets

        out[policy_idx] = new_payload
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--add", type=int, default=8, help="New input channels (6+1+1)")
    parser.add_argument("--new_mean", default=None, help="Comma-separated means for the new channels")
    parser.add_argument("--new_std", default=None, help="Comma-separated stds for the new channels")
    args = parser.parse_args()

    new_mean = [float(x) for x in args.new_mean.split(",")] if args.new_mean else DEFAULT_MEAN
    new_std = [float(x) for x in args.new_std.split(",")] if args.new_std else DEFAULT_STD

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    widened = widen(ckpt, int(args.add), new_mean, new_std)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(widened, out_path)

    old = ckpt[0]["model"]["a2c_network.rnn.rnn.weight_ih_l0"].shape
    new = widened[0]["model"]["a2c_network.rnn.rnn.weight_ih_l0"].shape
    print(f"[widen] rnn.weight_ih_l0 {tuple(old)} -> {tuple(new)}")
    print(f"[widen] running_mean {tuple(ckpt[0]['model']['running_mean_std.running_mean'].shape)} -> "
          f"{tuple(widened[0]['model']['running_mean_std.running_mean'].shape)}")
    print(f"[widen] wrote {out_path}")


if __name__ == "__main__":
    main()
