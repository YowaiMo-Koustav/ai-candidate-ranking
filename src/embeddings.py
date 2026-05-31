"""
embeddings.py — Generate text embeddings and compute semantic similarity.

Uses sentence-transformers for local embedding generation.
Supports batch processing for efficiency.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    """
    Load a sentence-transformers model for embedding generation.

    Args:
        model_name (str): Name of the sentence-transformers model.
                          Default is lightweight and fast.

    Returns:
        SentenceTransformer: Loaded model ready for encoding.

    Common models (speed vs quality tradeoff):
        - "all-MiniLM-L6-v2"       → Fast, good quality (recommended for hackathon)
        - "all-mpnet-base-v2"      → Best quality, slower
        - "paraphrase-MiniLM-L6-v2" → Good for paraphrase tasks
    """
    from sentence_transformers import SentenceTransformer

    print(f"⏳ Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"✅ Model loaded: {model_name} (dim={model.get_sentence_embedding_dimension()})")
    return model


def generate_embeddings(texts, model, batch_size=32, show_progress=True):
    """
    Generate embeddings for a list of texts.

    Args:
        texts (list[str]): List of text strings to embed.
        model (SentenceTransformer): Loaded sentence-transformers model.
        batch_size (int): Batch size for encoding.
        show_progress (bool): Show progress bar.

    Returns:
        np.ndarray: Matrix of shape (n_texts, embedding_dim).

    Example:
        >>> model = load_embedding_model()
        >>> texts = ["I am a data scientist", "Looking for ML engineer"]
        >>> embeddings = generate_embeddings(texts, model)
        >>> print(embeddings.shape)
        (2, 384)
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    return embeddings


def compute_similarity(job_embedding, candidate_embeddings):
    """
    Compute cosine similarity between one job and all candidates.

    Args:
        job_embedding (np.ndarray): Single job embedding, shape (1, dim) or (dim,).
        candidate_embeddings (np.ndarray): All candidate embeddings, shape (n, dim).

    Returns:
        np.ndarray: Similarity scores, shape (n,), values between -1 and 1.

    Example:
        >>> scores = compute_similarity(job_emb, candidate_embs)
        >>> print(scores.shape)
        (100,)
    """
    # Ensure job_embedding is 2D
    if job_embedding.ndim == 1:
        job_embedding = job_embedding.reshape(1, -1)

    similarities = cosine_similarity(job_embedding, candidate_embeddings)
    return similarities.flatten()


def compute_pairwise_similarity(embeddings_a, embeddings_b):
    """
    Compute pairwise cosine similarity between two sets of embeddings.

    Args:
        embeddings_a (np.ndarray): First set, shape (m, dim).
        embeddings_b (np.ndarray): Second set, shape (n, dim).

    Returns:
        np.ndarray: Similarity matrix, shape (m, n).
    """
    return cosine_similarity(embeddings_a, embeddings_b)
