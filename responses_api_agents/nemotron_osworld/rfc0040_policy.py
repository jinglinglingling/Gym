# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact RFC0040 K/S screenshot-retention policy for token-captured rollouts."""

from __future__ import annotations

from typing import Any, Mapping

from nemo_gym.context_management.history import (
    HistoryViewPlan,
    KeepPartRef,
    OmissionArtifact,
    PolicyDecisionRecord,
    SemanticHistory,
    _config_digest,
    _lineage_record,
    _ordered_id_digest,
    _semantic_part_digest,
    canonical_digest,
)


class RFC0040ImageHistoryPolicy:
    """Pin observation zero and retain the RFC0040 boundary/tail image window."""

    name = "rfc0040_k_s"
    version = "1"

    def __init__(self, config: Mapping[str, Any]):
        unknown = set(config) - {"block_k", "boundary_s", "omission_marker"}
        if unknown:
            raise ValueError(f"Unknown RFC0040 history setting(s): {sorted(unknown)}")
        self.block_k = int(config.get("block_k", 8))
        self.boundary_s = int(config.get("boundary_s", 3))
        self.omission_marker = str(
            config.get("omission_marker", "[Image omitted for text cleaning]")
        )
        if self.block_k < 1 or not 1 <= self.boundary_s <= self.block_k:
            raise ValueError("RFC0040 image history requires block_k >= boundary_s >= 1")
        self._config = {
            "block_k": self.block_k,
            "boundary_s": self.boundary_s,
            "omission_marker": self.omission_marker,
        }
        self._digest = _config_digest({"type": self.name, "config": self._config})

    def plan(self, history: SemanticHistory, *, decision_turn: int) -> HistoryViewPlan:
        image_parts = [(event, part) for event, part in history.parts if part.kind == "image"]
        image_count = len(image_parts)
        if image_count <= self.block_k:
            retained_image_ids = {part.part_id for _, part in image_parts}
        else:
            step_idx = image_count - 1
            boundary = (step_idx // self.block_k) * self.block_k
            window_start = max(1, boundary - (self.boundary_s - 1))
            retained_image_ids = {
                part.part_id
                for index, (_, part) in enumerate(image_parts)
                if index == 0 or index >= window_start
            }

        keep: list[KeepPartRef] = []
        artifacts: list[OmissionArtifact] = []
        disposition_by_part_id: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {}
        omitted_part_ids: list[str] = []
        protected_part_ids: list[str] = []

        for event, part in history.parts:
            source_digest = _semantic_part_digest(event, part)
            if part.kind != "image" or part.part_id in retained_image_ids:
                keep.append(KeepPartRef(event.event_id, part.part_id))
                disposition_by_part_id[part.part_id] = (
                    "kept",
                    (part.part_id,),
                    (source_digest,),
                )
                if part.kind == "image":
                    protected_part_ids.append(part.part_id)
                continue

            omitted_part_ids.append(part.part_id)
            artifact_id = f"omission-{self._digest[:12]}-{part.part_id}"
            artifact = OmissionArtifact(
                artifact_id=artifact_id,
                source_first_part_id=part.part_id,
                source_last_part_id=part.part_id,
                source_part_count=1,
                source_digest=_ordered_id_digest((part.part_id,)),
                text=self.omission_marker,
                anchor_part_id=part.part_id,
            )
            artifacts.append(artifact)
            disposition_by_part_id[part.part_id] = (
                "replaced",
                (artifact_id,),
                (
                    canonical_digest(
                        {
                            "type": "omission_marker",
                            "artifact_id": artifact_id,
                            "text": self.omission_marker,
                        }
                    ),
                ),
            )

        retained_ids = tuple(ref.part_id for ref in keep)
        decision = PolicyDecisionRecord(
            policy_name=self.name,
            policy_version=self.version,
            config_digest=self._digest,
            protected_part_ids=tuple(protected_part_ids),
            changed_part_ranges=tuple((part_id, part_id) for part_id in omitted_part_ids),
            retained_part_count=len(retained_ids),
            omitted_part_count=len(omitted_part_ids),
            selection_digest=_ordered_id_digest(
                (*retained_ids, "--omitted--", *omitted_part_ids)
            ),
            inserted_artifact_ids=tuple(artifact.artifact_id for artifact in artifacts),
            decision_turn=decision_turn,
            lineage=_lineage_record(
                history,
                transformation_type=self.name,
                transformation_version=self.version,
                configuration_digest=self._digest,
                disposition_by_part_id=disposition_by_part_id,
                lossy=bool(omitted_part_ids),
            ),
        )
        return HistoryViewPlan(
            keep=tuple(keep),
            artifacts=tuple(artifacts),
            decision=decision,
        )

