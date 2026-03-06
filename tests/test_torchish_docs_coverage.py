import re
from pathlib import Path

import pytest
import torch

from torch2jax import Torchish


def _extract_tensor_members_from_docs(docs_html: str) -> set[str]:
    """Extract documented Tensor member names from the downloaded docs HTML.

    We parse Sphinx-generated links of the form:
    generated/torch.Tensor.<member>.html
    """
    matches = re.findall(r"generated/torch\.Tensor\.([A-Za-z_][A-Za-z0-9_]*)\.html", docs_html)
    return set(matches)


def test_torchish_implements_documented_tensor_members():
    docs_path = Path(__file__).parent / "data" / "pytorch_tensors.html"
    if not docs_path.exists():
        pytest.skip(f"Missing docs fixture: {docs_path}")

    docs_html = docs_path.read_text(encoding="utf-8")
    documented_members = _extract_tensor_members_from_docs(docs_html)

    # Avoid doc/runtime version skew by restricting to members that exist on local torch.Tensor.
    runtime_documented_members = {
        name for name in documented_members if hasattr(torch.Tensor, name)
    }

    missing = sorted(
        name for name in runtime_documented_members if not hasattr(Torchish, name)
    )

    assert not missing, (
        f"Torchish is missing {len(missing)} documented torch.Tensor members. "
        f"First 50 missing: {missing[:50]}"
    )
