"""Mock embedder for testing that returns hash-based one-dimensional embeddings."""

from typing import List


class MockEmbedder:
    """Mock embedder that returns one-dimensional embeddings based on string hash.

    This provides a deterministic, lightweight embedding for testing without
    requiring actual embedding models or API calls.
    """

    def embed(self, text: str) -> List[float]:
        """Generate a one-dimensional embedding from text hash.

        Args:
            text: Input text to embed

        Returns:
            Single-element list containing normalized hash value
        """
        # Use Python's built-in hash, normalize to [-1, 1] range
        hash_value = hash(text)
        # Normalize: map hash to [-1, 1] using modulo and division
        normalized = (hash_value % 10000) / 5000.0 - 1.0
        return [normalized]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of input texts to embed

        Returns:
            List of one-dimensional embeddings
        """
        return [self.embed(text) for text in texts]

    @property
    def dimension(self) -> int:
        """Return embedding dimension (always 1 for mock)."""
        return 1

    @property
    def model_id(self) -> str:
        """Return mock model identifier."""
        return "mock-embedder-v1"
