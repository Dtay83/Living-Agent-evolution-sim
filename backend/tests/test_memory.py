from living_agent_v2.memory import MemoryStore, embed_text


def test_embedding_is_deterministic():
    assert embed_text("energy equals mass") == embed_text("energy equals mass")


def test_memory_search_returns_stored_record():
    store = MemoryStore()
    record = store.add("agent-1", "accepted", "energy equals 500")

    results = store.search("energy equals 500", limit=1)

    assert results[0].id == record.id

