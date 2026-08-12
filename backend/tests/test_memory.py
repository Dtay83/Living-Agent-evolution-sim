import pytest
from living_agent_v2.memory import cosine_similarity, embed_text


def test_embeddings_are_deterministic_and_normalized() -> None:
    first = embed_text("symbolic energy discovery")
    second = embed_text("symbolic energy discovery")

    assert first == second
    assert cosine_similarity(first, second) == pytest.approx(1.0)


def test_empty_embedding_has_zero_similarity() -> None:
    empty = embed_text("")

    assert cosine_similarity(empty, empty) == 0.0

