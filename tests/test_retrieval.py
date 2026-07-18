import json
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


class SourcePriorityEmbeddingModel:
    def encode(self, texts, **_kwargs):
        vectors = []
        for text in texts:
            lowered = text.casefold()
            if lowered.startswith("query:"):
                vectors.append([1.0, 0.0])
            elif "curated family" in lowered:
                vectors.append([0.95, 0.312])
            else:
                vectors.append([1.0, 0.0])
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


def test_preload_keeps_the_index_embedding_model_resident(tmp_path, monkeypatch):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "identity.md").write_text("Identity fact.", encoding="utf-8")
    config = SimpleNamespace(
        data=SimpleNamespace(cleaned=tmp_path / "missing.json"),
        retrieval=SimpleNamespace(
            database=tmp_path / "retrieval.sqlite3",
            knowledge_dir=knowledge,
            include_cleaned_telegram=False,
            chunk_chars=900,
            chunk_overlap_chars=120,
            embedding_model="resident-test-model",
            embedding_device="cpu",
            embedding_batch_size=8,
            vector_weight=0.75,
        ),
    )
    model = FakeEmbeddingModel()
    monkeypatch.setattr(retrieval, "_embedding_model", lambda *_args: model)
    retrieval._RESIDENT_EMBEDDING_MODELS.clear()
    retrieval.build_retrieval_index(config)

    metadata = retrieval.preload_retrieval_embedding_model(config.retrieval.database)

    assert metadata == {
        "embedding_model": "resident-test-model",
        "embedding_device": "cpu",
    }
    assert retrieval._RESIDENT_EMBEDDING_MODELS[
        ("resident-test-model", "cpu")
    ] is model


def test_raw_export_ingests_every_chat_type(tmp_path):
    export = tmp_path / "result.json"
    export.write_text(
        json.dumps(
            {
                "chats": {
                    "list": [
                        {
                            "id": 1,
                            "name": "Friend",
                            "type": "personal_chat",
                            "messages": [
                                {
                                    "from": "Friend",
                                    "date_unixtime": "100",
                                    "text": "personal message",
                                }
                            ],
                        },
                        {
                            "id": 2,
                            "name": "Saved Messages",
                            "type": "saved_messages",
                            "messages": [
                                {
                                    "from": "Rodion",
                                    "date_unixtime": "200",
                                    "text": "saved note",
                                }
                            ],
                        },
                        {
                            "id": 3,
                            "name": "Media-only group",
                            "type": "private_group",
                            "messages": [],
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    documents = list(retrieval._raw_telegram_documents(export))

    assert [source for source, _ in documents] == [
        "telegram/personal_chat/1/session_0000",
        "telegram/saved_messages/2/session_0000",
        "telegram/private_group/3/session_0000",
    ]
    assert "personal message" in documents[0][1]
    assert "saved note" in documents[1][1]
    assert "Media-only group" in documents[2][1]


def test_curated_knowledge_is_reserved_ahead_of_similar_conversation_noise(
    tmp_path,
    monkeypatch,
):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "identity.md").write_text(
        "Curated family facts about the owner.",
        encoding="utf-8",
    )
    cleaned = tmp_path / "cleaned.json"
    cleaned.write_text(
        json.dumps(
            {
                "chats": [
                    {
                        "id": 1,
                        "name": "noise",
                        "type": "personal_chat",
                        "sessions": [
                            {
                                "session_id": "session-1",
                                "messages": [
                                    {
                                        "from": "other",
                                        "text": "Unrelated conversation family noise.",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        data=SimpleNamespace(cleaned=cleaned),
        retrieval=SimpleNamespace(
            database=tmp_path / "retrieval.sqlite3",
            knowledge_dir=knowledge,
            include_cleaned_telegram=True,
            chunk_chars=900,
            chunk_overlap_chars=120,
            embedding_model="source-priority-embeddings",
            embedding_device="cpu",
            embedding_batch_size=8,
            vector_weight=0.75,
        ),
    )
    monkeypatch.setattr(
        retrieval,
        "_embedding_model",
        lambda *_args: SourcePriorityEmbeddingModel(),
    )

    retrieval.build_retrieval_index(config)
    results = retrieval.search_retrieval(config.retrieval.database, "family", 2)

    assert results[0]["source"] == "knowledge/identity.md#0"
    assert results[0]["source_priority"] == "curated_knowledge"
    assert results[1]["source_priority"] == "conversation_memory"
