"""Per-epoch trace, for the convergence probe described in project_status.md §9.

The reported runs only keep the *selected* epoch, which is enough to compare
models but not to answer two questions the status doc left open:

  1. Where does the composite selection score actually flatten? ``best_epoch``
     sat at 34-49 against a 50-epoch budget for all three models, so the budget
     may simply have been too small.
  2. Is the routing matrix ``W`` still growing when training stops? It is small
     in absolute terms (mean |W| 0.0195-0.0374) but that is 2.4-4.7x its
     initialisation (``0.01 * randn`` -> mean |W| = 0.01*sqrt(2/pi) = 0.00798),
     so "stuck at init" is the wrong reading. Still climbing means the budget is
     the problem; flat means training found no reason to grow it further, and
     only then is an init-scale experiment a controlled one.

One JSONL row per epoch answers both at once, at no extra training cost.
"""
import json
import math
import os

# mean |x| of a half-normal, i.e. what 0.01 * randn starts at.
INIT_W_ABS_MEAN = 0.01 * math.sqrt(2.0 / math.pi)


def routing_w_abs_mean(model):
    """Mean |W| of the routing transform, or None for models without one.

    Phase2 has no routing layer, and the probe runs it too -- the budget
    question applies to all three models, the W question only to two.
    """
    routing = getattr(model, "routing", None)
    if routing is None or not hasattr(routing, "W"):
        return None
    return float(routing.W.detach().abs().mean())


def append(path, row):
    """Append one row. Opened per write so a killed run keeps its trace.

    These probes run for hours unattended and have been cut short before by
    the machine sleeping, so buffering the trace until exit is not safe.
    """
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def trace_path(out_dir, tag, fold):
    if not os.environ.get("TRACE"):
        return None
    return os.path.join(out_dir, "trace", f"{tag}_fold{fold}.jsonl")


def only_folds():
    """Folds to actually train, from ONLY_FOLDS=0 or ONLY_FOLDS=0,2.

    The probe trains one fold out of five. Narrowing ``n_folds`` instead would
    change the split, and the probe has to see the same split as the runs it is
    meant to explain.
    """
    v = os.environ.get("ONLY_FOLDS")
    if not v:
        return None
    return {int(x) for x in v.replace(" ", "").split(",") if x != ""}
