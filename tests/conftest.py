"""The pod's VLM writes its (fake) vLLM log to a temporary folder in the tests, not runpod/vllm.log."""

import os
import tempfile

os.environ.setdefault("VLLM_LOG", os.path.join(tempfile.mkdtemp(prefix="vllm-"), "vllm.log"))
