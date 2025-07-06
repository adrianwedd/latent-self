import base64
import io
import numpy as np
import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def latent_directions():
    """Load latent direction vectors from the archived models."""
    path = Path("models/latent_directions.npz.txt")
    data = base64.b64decode(path.read_bytes())
    with np.load(io.BytesIO(data)) as npz:
        return {k: npz[k].ravel() for k in npz.files}
