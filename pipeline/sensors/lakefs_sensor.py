"""LakeFS-backed Dagster sensors."""

from __future__ import annotations

from dagster import (
    DefaultSensorStatus,
    JobDefinition,
    RunRequest,
    SensorEvaluationContext,
    sensor,
)

from ..resources import LakeFSResource


def build_lakefs_sensor(target_job: JobDefinition):
    @sensor(
        name="lakefs_new_commit_sensor",
        job=target_job,
        minimum_interval_seconds=300,
        default_status=DefaultSensorStatus.STOPPED,
        required_resource_keys={"lakefs"},
    )
    def _sensor(context: SensorEvaluationContext):
        lakefs: LakeFSResource = context.resources.lakefs
        commits = lakefs.list_recent_commits(amount=20)
        if not commits:
            context.log.info("No LakeFS commits returned; sensor idle")
            return

        cursor = context.cursor
        new_commits: list[dict[str, str]] = []
        for commit in commits:
            if cursor and commit["id"] == cursor:
                break
            new_commits.append(commit)

        if not new_commits:
            context.log.debug("No unseen commits. Latest cursor=%s", cursor)
            return

        for commit in reversed(new_commits):
            yield RunRequest(
                run_key=commit["id"],
                run_config={},
                tags={"lakefs_commit": commit["id"], "lakefs_message": commit.get("message", "")},
            )

        context.update_cursor(new_commits[0]["id"])

    return _sensor


__all__ = ["build_lakefs_sensor"]
