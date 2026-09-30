# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Print every installed distribution as `name==version`, for use as a pip/uv constraints file.

The image installs scipy, the one package the decisio wheel adds, with these as constraints, so that scipy's own
dependencies cannot replace anything the vLLM image ships (torch, numpy, transformers); a conflict fails the build.
"""

import importlib.metadata as metadata

pins = {f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata["Name"]}
print("\n".join(sorted(pins, key=str.lower)))
