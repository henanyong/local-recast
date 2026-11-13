"""Shared write lock for character ``meta.json`` (N36).

One module-level :class:`threading.RLock`, used by **both** sides that
write ``meta.json`` so they can never write past each other:

* the GUI thread — every GUI-side meta write goes through
  ``recast.core.characters.save_meta``, which takes this lock;
* the 3D-generation worker thread — ``recast.desktop.model3d_queue``
  holds this lock around its ``generate_fn`` call, because the real
  registration (``service.generate_model_3d`` ->
  ``generate_model_glb`` -> ``save_meta`` / ``set_assembly``) happens
  inside that call and cannot be split out from the queue side.

It must be an :class:`RLock`, not a plain ``Lock``: the worker holds it
while generating, and the registration path calls ``save_meta`` again
from the same worker thread — a plain lock would deadlock there.

SQLite has its own write serialisation, so this lock is only about
``meta.json`` files and the read-modify-write critical sections around
them. Do not invent a second lock for the same files elsewhere; import
this one.

Contract for read-modify-write: ``save_meta`` alone only serialises
the write itself (no interleaving, no torn file). A caller that does
``load_meta`` -> mutate -> ``save_meta`` must hold this lock across
the *whole* span — otherwise it can load a stale copy, another thread
saves, and its save then overwrites the other thread's section with
the old value. Hold it like this::

    with meta_write_lock():
        meta = load_meta(folder) or default_meta(name)
        meta["voice"] = ...
        save_meta(folder, meta)

(``save_meta`` re-enters the RLock, so nesting is fine.) The 3D worker
holds it around its whole ``generate_fn`` call for the same reason:
the registration inside cannot be split out from the queue side.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

# The single shared lock for all meta.json writes. Import it, never copy it.
META_WRITE_LOCK = threading.RLock()


@contextmanager
def meta_write_lock() -> Iterator[None]:
    """Hold the shared meta.json write lock (re-entrant, same thread)."""
    with META_WRITE_LOCK:
        yield
