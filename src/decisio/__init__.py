# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""decisio: a prefill-only decision server on vLLM.

  decisio.serve         the HTTP server (`python -m decisio.serve.vllm_engine`), its routes, tasks and engines
  decisio.readout       the letters readout and the per-task corrections (calibration, intent head)
  decisio.vllm_plugin   the model classes vLLM loads through the `vllm.general_plugins` entry point
  decisio.bench         the benchmark stage: JevBench v1.5 open-set scoring, Decision Index reports, latency

Importing `decisio` pulls in nothing beyond the standard library.
"""

__version__ = "0.11.0"
