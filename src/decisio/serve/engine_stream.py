# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The CUDA stream an in-process vLLM engine does its work on, kept the same in every thread that calls it.

vLLM 0.31.0 tracks "the current stream" per thread (`vllm.utils.torch_utils.current_stream`: a thread-local, made on a
thread's first use) and keys some persistent GPU resources by it. The Marlin kernels (the W4A16 compressed-tensors
checkpoints, GPTQ, AWQ; the linear layers and the Marlin MoE) keep their lock buffer in the workspace manager under
`("marlin_locks", current_stream())`; once the warm-up has finished the manager is locked, and a key it has not seen
raises `AssertionError: Workspace is locked but (... 'marlin_locks' ...) was not allocated during warmup for the current
ubatch and lane`.

decisio builds the engine and warms it up on the main thread, then serves each request on a thread of FastAPI's pool and
confirms a failed request with a probe on a thread of its own. Each of those threads has no stream yet, makes one, and
asks for a lock buffer under a key the warm-up never made: the first request of a W4A16 checkpoint ended the engine
(Lab 2's card session of 2026-10-15; plain vLLM called from one thread answered). Binding every calling thread to the
stream the engine was built on gives them the key the warm-up allocated. The engine takes one request at a time
(`LettersEngine._lock`), so the threads never use the stream together.

    stream = capture()      on the thread that built and warmed the engine
    bind(stream)            at the start of any call into the engine, on any thread
"""

from __future__ import annotations


def capture():
    """The calling thread's current stream as vLLM tracks it (made if the thread has none), or None where there is no
    vLLM or no CUDA (the CPU stand-in and the MLX backend need nothing)."""
    try:
        from vllm.utils.torch_utils import current_stream

        return current_stream()
    except Exception:  # noqa: BLE001 - no vLLM, no CUDA: nothing to bind
        return None


def bind(stream, set_stream=None) -> None:
    """Make `stream` the calling thread's current stream. vLLM replaces `torch.cuda.set_stream` with a version that also
    records the stream in its thread-local, which is what its `current_stream()` reads. `set_stream` stands in for
    tests."""
    if stream is None:
        return
    if set_stream is None:
        import torch

        set_stream = torch.cuda.set_stream
    set_stream(stream)
