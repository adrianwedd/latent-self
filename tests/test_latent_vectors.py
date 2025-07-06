import numpy as np

NEW_DIRS = [
    'beauty',
    'happy',
    'angry',
    'sad',
    'fear',
    'disgust',
    'surprise',
]

BASE_DIRS = ['age', 'gender', 'smile']


def test_vectors_present(latent_directions):
    for key in NEW_DIRS:
        assert key in latent_directions, f"{key} missing from latent_directions.npz"


def test_vector_norms_and_orthogonality(latent_directions):
    for key in NEW_DIRS:
        vec = latent_directions[key]
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, atol=1e-3), f"{key} norm {norm}"
        for base in BASE_DIRS:
            if base in latent_directions:
                dot = abs(float(np.dot(vec, latent_directions[base])))
                assert dot < 0.3, f"{key} not orthogonal to {base}: {dot}"
