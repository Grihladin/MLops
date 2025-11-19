"""Resource helpers for talking to LakeFS."""

from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath
from typing import Iterable

import requests
from dagster import ConfigurableResource, get_dagster_logger

try:
    from lakefs_spec import LakeFSFileSystem
except ImportError as exc:  # pragma: no cover - configuration issue
    raise RuntimeError(
        "lakefs-spec must be installed to use the LakeFSResource"
    ) from exc

try:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
except IndexError:  # pragma: no cover - safety net if layout changes
    _REPO_ROOT = Path(__file__).resolve().parent


class LakeFSResource(ConfigurableResource):
    """Minimal LakeFS helper built on top of lakefs-spec."""

    endpoint_url: str | None = None
    repo: str | None = None
    branch: str = "main"
    access_key: str | None = None
    secret_key: str | None = None
    prefix: str = "real_data"
    fallback_local_path: str | None = None
    cache_subdir: str = "lakefs_cache"
    request_timeout: int = 20

    def _has_remote_config(self) -> bool:
        return bool(self.endpoint_url and self.repo and self.branch)

    def _build_fs(self) -> LakeFSFileSystem:
        if not self._has_remote_config():
            raise RuntimeError(
                "LakeFS connection details are missing. Provide endpoint_url, repo, and branch"
            )
        return LakeFSFileSystem(
            repo=self.repo,
            reference=self.branch,
            endpoint=self.endpoint_url,
            username=self.access_key,
            password=self.secret_key,
        )

    @property
    def logger(self):
        return get_dagster_logger()

    def _copy_local_tree(self, source: Path, destination: Path) -> None:
        if not source.exists():
            raise FileNotFoundError(f"Local LakeFS fallback path {source} was not found")
        for file_path in source.rglob("*"):
            if file_path.is_dir():
                continue
            rel = file_path.relative_to(source)
            dest = destination / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest)

    def _fallback_source(self, prefix: str) -> Path | None:
        candidates: list[Path] = []
        if self.fallback_local_path:
            candidates.append(Path(self.fallback_local_path).expanduser().resolve())
        repo_root = _REPO_ROOT.resolve()
        if repo_root not in candidates:
            candidates.append(repo_root)

        for root in candidates:
            candidate = (root / prefix).expanduser().resolve()
            if candidate.exists():
                return candidate
        return None

    def _download_from_remote(self, prefix: str, destination: Path) -> tuple[Path, str | None]:
        fs = self._build_fs()
        posix_prefix = PurePosixPath(prefix)
        listings = fs.find(str(posix_prefix))
        if isinstance(listings, dict):
            iterator: Iterable[str] = listings.keys()
        else:
            iterator = listings

        copied_files = 0
        for object_path in iterator:
            object_posix = PurePosixPath(object_path)
            if object_posix.as_posix().endswith("/"):
                continue
            try:
                rel = object_posix.relative_to(posix_prefix)
            except ValueError:
                rel = object_posix.name
            target = destination / Path(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            with fs.open(object_posix.as_posix(), "rb") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            copied_files += 1

        if not copied_files:
            raise FileNotFoundError(
                f"No objects were copied from prefix '{prefix}'. Check the repo/branch configuration."
            )
        commit_id = self.resolve_branch_commit()
        return destination, commit_id

    def download_prefix(self, prefix: str, destination: Path) -> tuple[Path, str | None]:
        destination = destination.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        local_source = self._fallback_source(prefix)

        if not self._has_remote_config():
            if local_source:
                self.logger.info(
                    "LakeFS config missing; falling back to local data at %s", local_source
                )
                self._copy_local_tree(local_source, destination)
                return destination, None
            raise RuntimeError(
                "LakeFS connection details are missing and no local fallback data was found."
            )

        try:
            return self._download_from_remote(prefix, destination)
        except Exception as exc:
            if local_source:
                self.logger.warning(
                    "Failed to download from LakeFS (%s). Falling back to %s", exc, local_source
                )
                self._copy_local_tree(local_source, destination)
                return destination, None
            raise

    def resolve_branch_commit(self) -> str | None:
        if not (self.endpoint_url and self.repo and self.branch):
            return None
        url = f"{self.endpoint_url.rstrip('/')}/api/v1/repositories/{self.repo}/branches/{self.branch}"
        auth = (self.access_key or "", self.secret_key or "")
        response = requests.get(url, auth=auth, timeout=self.request_timeout)
        if not response.ok:
            self.logger.warning(
                "Failed to resolve branch commit for %s (%s)", self.branch, response.text
            )
            return None
        payload = response.json()
        return payload.get("commit_id") or payload.get("id")

    def list_recent_commits(self, amount: int = 5) -> list[dict[str, str]]:
        if not (self.endpoint_url and self.repo):
            return []
        params = {"amount": amount, "ref": self.branch}
        url = f"{self.endpoint_url.rstrip('/')}/api/v1/repositories/{self.repo}/commits"
        auth = (self.access_key or "", self.secret_key or "")
        response = requests.get(url, params=params, auth=auth, timeout=self.request_timeout)
        if not response.ok:
            self.logger.warning("Could not list LakeFS commits: %s", response.text)
            return []
        payload = response.json()
        results = payload.get("results") or payload.get("commits") or []
        commits: list[dict[str, str]] = []
        for item in results:
            commits.append(
                {
                    "id": item.get("id") or item.get("commit_id", ""),
                    "message": item.get("message", ""),
                    "creation_date": item.get("creation_date") or item.get("creation_time"),
                }
            )
        return commits

    def download_default_prefix(self, destination: Path) -> tuple[Path, str | None]:
        return self.download_prefix(self.prefix, destination)
