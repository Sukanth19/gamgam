"""ChromaDB wrapper. We compute embeddings ourselves (Gemini) and hand Chroma
the raw vectors, so Chroma is used purely as a vector index + metadata store."""

from pathlib import Path

import chromadb

from app import config


class GameVectorStore:
    def __init__(self, path: Path | str | None = config.CHROMA_DIR,
                 collection: str = config.CHROMA_COLLECTION):
        # path=None -> in-memory (for tests)
        self._client = (chromadb.EphemeralClient() if path is None
                        else chromadb.PersistentClient(path=str(path)))
        # Cosine distance: we care about direction (meaning), not magnitude.
        self.col = self._client.get_or_create_collection(
            collection, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None)

    def existing_hashes(self) -> dict[int, str]:
        """game_id -> profile_hash for everything already embedded."""
        got = self.col.get(include=["metadatas"])
        return {int(i): (m or {}).get("profile_hash", "") for i, m in zip(got["ids"], got["metadatas"])}

    def upsert(self, games: list[dict], vectors: list[list[float]], hashes: list[str]):
        self.col.upsert(
            ids=[str(g["id"]) for g in games],
            embeddings=vectors,
            # Chroma metadata must be scalar: lists become comma-joined strings.
            metadatas=[{
                "name": g["name"],
                "genres": ", ".join(g.get("genres") or []),
                "year": int(g["released"][:4]) if g.get("released") else 0,
                "rating": float(g.get("rating") or 0.0),
                "profile_hash": h,
            } for g, h in zip(games, hashes)],
        )

    def get_vector(self, game_id: int) -> list[float] | None:
        got = self.col.get(ids=[str(game_id)], include=["embeddings"])
        return list(got["embeddings"][0]) if got["ids"] else None

    def query(self, vector: list[float], k: int) -> list[dict]:
        """Nearest neighbours; similarity = 1 - cosine distance."""
        res = self.col.query(query_embeddings=[vector], n_results=k,
                             include=["metadatas", "distances"])
        return [{"id": int(i), **m, "similarity": 1.0 - d}
                for i, m, d in zip(res["ids"][0], res["metadatas"][0], res["distances"][0])]

    def get_vectors(self, game_ids: list[int]) -> dict[int, list[float]]:
        if not game_ids:
            return {}
        got = self.col.get(ids=[str(i) for i in game_ids], include=["embeddings"])
        return {int(i): list(v) for i, v in zip(got["ids"], got["embeddings"])}

    def query_with_vectors(self, vector: list[float], k: int) -> list[dict]:
        """Like query(), but also returns each hit's vector (needed for MMR)."""
        res = self.col.query(query_embeddings=[vector], n_results=k,
                             include=["metadatas", "distances", "embeddings"])
        return [{"id": int(i), **m, "similarity": 1.0 - d, "vector": list(v)}
                for i, m, d, v in zip(res["ids"][0], res["metadatas"][0],
                                      res["distances"][0], res["embeddings"][0])]

    def count(self) -> int:
        return self.col.count()


class LoreVectorStore:
    """Lore chunks. Metadata carries everything needed to filter (game,
    spoiler) and cite (title, section, url) without another DB lookup."""

    def __init__(self, path: Path | str | None = config.CHROMA_DIR,
                 collection: str = config.LORE_COLLECTION):
        self._client = (chromadb.EphemeralClient() if path is None
                        else chromadb.PersistentClient(path=str(path)))
        self.col = self._client.get_or_create_collection(
            collection, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None)

    def existing_hashes(self) -> dict[str, str]:
        got = self.col.get(include=["metadatas"])
        return {i: (m or {}).get("hash", "") for i, m in zip(got["ids"], got["metadatas"])}

    def delete(self, ids: list[str]):
        for start in range(0, len(ids), 500):
            self.col.delete(ids=ids[start:start + 500])

    def upsert(self, chunks: list[dict], vectors: list[list[float]], hashes: list[str]):
        self.col.upsert(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=vectors,
            documents=[c["text"] for c in chunks],
            metadatas=[{
                "game_id": c["game_id"], "game": c["game"], "title": c["title"],
                "section": c["section"], "url": c["url"] or "", "kind": c["kind"],
                "spoiler": bool(c["spoiler"] or c["page_spoiler"]), "hash": h,
            } for c, h in zip(chunks, hashes)],
        )

    def search(self, vector: list[float], k: int, game_id: int | None = None) -> list[dict]:
        res = self.col.query(query_embeddings=[vector], n_results=k,
                             where={"game_id": game_id} if game_id is not None else None,
                             include=["metadatas", "documents", "distances"])
        return [{"chunk_id": i, **m, "text": doc, "similarity": 1.0 - d}
                for i, m, doc, d in zip(res["ids"][0], res["metadatas"][0],
                                        res["documents"][0], res["distances"][0])]

    def count(self) -> int:
        return self.col.count()
