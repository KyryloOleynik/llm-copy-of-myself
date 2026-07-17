from types import SimpleNamespace

import numpy as np

from personal_ai import retrieval


class FakeEmbeddingModel:
    def encode(self, texts, **_kwargs):
        vectors = []
        for text in texts:
            lowered = text.casefold()
            if {"automobile", "vehicle", "car"} & set(lowered.split()):
                vectors.append([1.0, 0.0])
            elif "banana" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        values = np.asarray(vectors, dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_vector_search_finds_semantic_match_without_shared_keyword(tmp_path, monkeypatch):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "vehicle.md").write_text(
        "My first vehicle was a blue hatchback.",
        encoding="utf-8",
    )
    (knowledge / "food.md").write_text(
        "My favorite fruit is a banana.",
        encoding="utf-8",
    )
    config = SimpleNamespace(
        data=SimpleNamespace(cleaned=tmp_path / "missing.json"),
        retrieval=SimpleNamespace(
            database=tmp_path / "retrieval.sqlite3",
            knowledge_dir=knowledge,
            include_cleaned_telegram=False,
            chunk_chars=900,
            chunk_overlap_chars=120,
            embedding_model="fake-embeddings",
            embedding_device="cpu",
            embedding_batch_size=8,
            vector_weight=0.75,
        ),
    )
    monkeypatch.setattr(retrieval, "_embedding_model", lambda *_args: FakeEmbeddingModel())

    stats = retrieval.build_retrieval_index(config)
    results = retrieval.search_retrieval(config.retrieval.database, "automobile", 1)

    assert stats == {"documents": 2, "chunks": 2, "embedding_dimensions": 2}
    assert results[0]["source"] == "knowledge/vehicle.md#0"
    assert results[0]["keyword_score"] is None
    assert results[0]["retrieval"] == "hybrid_vector_keyword"
