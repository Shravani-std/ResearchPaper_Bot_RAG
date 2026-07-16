from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

from core.config import settings


class QdrantStore:
    def __init__(self) -> None:
        self.collection_name = settings.COLLECTION_NAME or "AIResearch_Bot"
        self._storage_path = Path(__file__).resolve().parent.parent / "vector_db" / "documents.json"
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._points: List[dict[str, Any]] = self._load_points()

    def create_collection(self) -> None:
        return None

    def upload_documents(self, documents: List[dict[str, Any]]) -> None:
        for doc in documents:
            self._points.append(
                {
                    "chunk_id": doc.get("chunk_id"),
                    "embedding": doc.get("embedding", []),
                    "payload": {
                                "text": doc["text"],
                                "source": doc["source"],
                                "page": doc["page"],
                                "section": doc["section"],
                                "chunk_id": doc["chunk_id"],
                            
                    },
                }
            )
        self._save_points()

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Any]:
        if not self._points:
            return []

        scored_points = []
        for point in self._points:
            similarity = self._cosine_similarity(query_vector, point["embedding"])
            scored_points.append((similarity, point))

        scored_points.sort(key=lambda item: item[0], reverse=True)
        results = []
        for _, point in scored_points[:top_k]:
            results.append(
                SimpleNamespace(
                    payload=point["payload"],
                    score=point.get("score", 0.0),
                )
            )

        return results

    def _save_points(self) -> None:
        with self._storage_path.open("w", encoding="utf-8") as handle:
            json.dump(self._points, handle, indent=2)

    def _load_points(self) -> List[dict[str, Any]]:
        if not self._storage_path.exists():
            return []
        with self._storage_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _cosine_similarity(left: List[float], right: List[float]) -> float:
        if not left or not right:
            return 0.0

        left_length = math.sqrt(sum(value * value for value in left))
        right_length = math.sqrt(sum(value * value for value in right))
        if left_length == 0 or right_length == 0:
            return 0.0

        dot_product = sum(a * b for a, b in zip(left, right))
        return dot_product / (left_length * right_length)