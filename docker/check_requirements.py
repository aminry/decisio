# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Check that every requirement of `decisio[serve]` is satisfied by what is installed, and say which is not.

The image installs the wheel with --no-deps on top of the vLLM image, because the resolver would re-read vLLM's own
metadata (which declares conflicting pins for one package) and could replace torch or transformers. This is the check
that resolution would have made: the wheel's requirements against the versions the image ships.
"""

import importlib.metadata as metadata
import sys

from packaging.requirements import Requirement

bad = []
for text in metadata.requires("decisio") or []:
    req = Requirement(text)
    if req.marker is not None and not any(req.marker.evaluate({"extra": extra}) for extra in ("", "serve")):
        continue
    try:
        have = metadata.version(req.name)
    except metadata.PackageNotFoundError:
        bad.append(f"{req.name}: required {req.specifier or 'any version'}, not installed")
        continue
    if not req.specifier.contains(have, prereleases=True):
        bad.append(f"{req.name}: required {req.specifier}, the image has {have}")
if bad:
    print("decisio's requirements are not met by the vLLM image:\n  " + "\n  ".join(bad), file=sys.stderr)
    sys.exit(1)
print("decisio[serve]: every requirement is met by the image")
