# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Persistence for elite and frontier archives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from esn.core.archives import EliteArchive, FrontierArchive
from esn.core.models import CandidateRecord


class ArchiveStore:
    """JSON-based save/load for EliteArchive and FrontierArchive."""

    @staticmethod
    def save_elite(archive: EliteArchive, path: Path) -> None:
        data = [c.model_dump(mode="json") for c in archive.get_all()]
        path.write_text(json.dumps(data, indent=2))

    @staticmethod
    def load_elite(path: Path, max_size: int = 50) -> EliteArchive:
        archive = EliteArchive(max_size=max_size)
        if not path.exists():
            return archive
        data: list[dict[str, Any]] = json.loads(path.read_text())
        for item in data:
            archive.insert(CandidateRecord.model_validate(item))
        return archive

    @staticmethod
    def save_frontier(archive: FrontierArchive, path: Path) -> None:
        novelty = archive.novelty_scores
        repairability = archive.repairability_scores
        entries = []
        for c in archive.get_all():
            entries.append(
                {
                    "candidate": c.model_dump(mode="json"),
                    "novelty": novelty.get(c.id, 0.0),
                    "repairability": repairability.get(c.id, 0.0),
                }
            )
        path.write_text(json.dumps(entries, indent=2))

    @staticmethod
    def load_frontier(
        path: Path,
        max_size: int = 100,
        novelty_threshold: float = 0.1,
        repairability_threshold: float = 0.1,
    ) -> FrontierArchive:
        archive = FrontierArchive(
            max_size=max_size,
            novelty_threshold=novelty_threshold,
            repairability_threshold=repairability_threshold,
        )
        if not path.exists():
            return archive
        entries: list[dict[str, Any]] = json.loads(path.read_text())
        for entry in entries:
            candidate = CandidateRecord.model_validate(entry["candidate"])
            archive.insert(
                candidate,
                novelty=entry.get("novelty", 0.0),
                repairability=entry.get("repairability", 0.0),
            )
        return archive

    @staticmethod
    def save_object_store(store: dict[str, Any], path: Path, search_object_class: type) -> None:
        """Serialize and persist the object store as {id: serialized_data}."""
        data: dict[str, str] = {}
        for obj_id, obj in store.items():
            serialized = obj.serialize()
            # serialize() returns str | bytes; store as str
            data[obj_id] = serialized if isinstance(serialized, str) else serialized.decode("utf-8")
        path.write_text(json.dumps(data, indent=2))

    @staticmethod
    def load_object_store(path: Path, search_object_class: type) -> dict[str, Any]:
        """Deserialize the object store from JSON."""
        if not path.exists():
            return {}
        data: dict[str, str] = json.loads(path.read_text())
        store: dict[str, Any] = {}
        for obj_id, serialized in data.items():
            store[obj_id] = search_object_class.deserialize(serialized)
        return store
