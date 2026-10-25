"""Who has the pod's GPU. A training job and vLLM never run at the same time (vLLM takes most of the GPU's memory),
and the RF-DETR labeller lets go of its model before vLLM starts. The services run in one process (server.py), so
they tell each other here instead of importing each other."""

_busy = {}  # holder ("training", "vlm") -> callable: a reason it holds the GPU now, or None
_release = []  # callables that free GPU memory a service only caches (the labeller's loaded models)


def holder(name, busy):
    _busy[name] = busy


def on_release(fn):
    _release.append(fn)


def busy(*names):
    """The first reason one of these holders has the GPU now, or None."""
    for n in names:
        reason = _busy.get(n, lambda: None)()
        if reason:
            return reason
    return None


def release():
    for fn in _release:
        fn()
