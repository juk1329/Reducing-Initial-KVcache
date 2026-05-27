# ------------------------------------------------------------------------------
# Reducing-Initial-Cache (RIC)
# Streaming KV-cache compaction that bounds *peak* prefill memory.
# Built on top of KVzip (https://github.com/snu-mllab/KVzip); we reuse its model
# wrapper, reconstruction-based importance scoring, and eviction machinery.
# ------------------------------------------------------------------------------
import os
import sys

# KVzip lives next to this package. Add it to sys.path so we can reuse its code
# WITHOUT installing it (its pyproject pins torch==2.3.0 / numpy==1.26 which would
# downgrade the env). See memory note `ric-env-setup-done`.
_HERE = os.path.dirname(os.path.abspath(__file__))
KVZIP_DIR = os.environ.get(
    "KVZIP_DIR", os.path.normpath(os.path.join(_HERE, "..", "..", "KVzip"))
)
if KVZIP_DIR not in sys.path:
    sys.path.insert(0, KVZIP_DIR)

# sanity: KVzip code uses top-level package imports like `from attention.kvcache import ...`
if not os.path.isdir(os.path.join(KVZIP_DIR, "attention")):
    raise RuntimeError(
        f"KVzip not found at {KVZIP_DIR!r}. Set KVZIP_DIR env var to the KVzip checkout."
    )

__all__ = ["KVZIP_DIR"]
