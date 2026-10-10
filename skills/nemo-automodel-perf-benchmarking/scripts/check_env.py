# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Print the package versions a container resolves and import the training entry points.

Usage: python check_env.py <model.module:ClassName> [more.module:Class ...]

Run this on one GPU after every baseline change and before any multi-node job.
"""

import importlib
import sys

import torch
import transformers

print("python", sys.version.split()[0], "torch", torch.__version__)
for name in ("huggingface_hub", "tokenizers", "triton"):
    try:
        module = importlib.import_module(name)
        print(name, getattr(module, "__version__", "?"), module.__file__)
    except ImportError as exc:
        print(name, "MISSING", exc)
print("transformers", transformers.__version__, transformers.__file__)
import nemo_automodel._transformers.auto_config  # noqa: E402,F401
import nemo_automodel.recipes.llm.benchmark  # noqa: E402,F401

for spec in sys.argv[1:]:
    module_name, _, class_name = spec.partition(":")
    getattr(importlib.import_module(module_name), class_name)
    print("import ok", spec)
print("ENV OK")
