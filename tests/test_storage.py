from __future__ import annotations

from pathlib import Path

import pytest

from dkenergy_forecast.storage import ArtifactStore, join_uri, parse_uri


class FakeS3Error(Exception):
    def __init__(self, code: str) -> None:
        self.response = {"Error": {"Code": code}}


class FailingS3Client:
    def __init__(self, code: str) -> None:
        self.code = code

    def download_file(self, bucket: str, key: str, destination: str) -> None:
        raise FakeS3Error(self.code)

    def head_object(self, *, Bucket: str, Key: str) -> None:
        raise FakeS3Error(self.code)


def test_parse_uri_supports_local_file_and_s3() -> None:
    local = parse_uri("data/model_ready/panel.parquet")
    file_uri = parse_uri("file:///tmp/panel.parquet")
    s3_uri = parse_uri("s3://bucket/prefix/latest/manifest.json")

    assert local.is_local
    assert local.path == "data/model_ready/panel.parquet"
    assert file_uri.is_local
    assert file_uri.path == "/tmp/panel.parquet"
    assert s3_uri.is_s3
    assert s3_uri.bucket == "bucket"
    assert s3_uri.key == "prefix/latest/manifest.json"


def test_join_uri_preserves_s3_and_local_roots() -> None:
    assert join_uri("s3://bucket/prefix", "latest", "manifest.json") == "s3://bucket/prefix/latest/manifest.json"
    assert join_uri("/tmp/store", "latest", "manifest.json") == str(Path("/tmp/store/latest/manifest.json"))


def test_local_artifact_store_uploads_and_downloads_prefixes(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "store")
    source = tmp_path / "source"
    source.mkdir()
    (source / "nested").mkdir()
    (source / "nested" / "artifact.txt").write_text("ok", encoding="utf-8")

    uploaded = store.upload_prefix(source, "state/data")
    destination = tmp_path / "downloaded"
    downloaded = store.download_prefix("state/data", destination, required=True)

    assert uploaded == ["state/data/nested/artifact.txt"]
    assert [path.relative_to(destination).as_posix() for path in downloaded] == ["nested/artifact.txt"]
    assert (destination / "nested" / "artifact.txt").read_text(encoding="utf-8") == "ok"


def test_upload_prefix_can_preserve_append_only_objects(tmp_path) -> None:
    store_root = tmp_path / "store"
    existing = store_root / "raw" / "existing.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("original", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    (source / "existing.json").write_text("replacement", encoding="utf-8")
    (source / "new.json").write_text("new", encoding="utf-8")

    uploaded = ArtifactStore(store_root).upload_prefix(
        source,
        "raw",
        skip_existing=True,
    )

    assert uploaded == ["raw/new.json"]
    assert existing.read_text(encoding="utf-8") == "original"


def test_optional_s3_download_ignores_only_missing_objects(
    tmp_path, monkeypatch
) -> None:
    store = ArtifactStore("s3://bucket/prefix")
    destination = tmp_path / "history.parquet"
    monkeypatch.setattr(
        "dkenergy_forecast.storage._s3_client",
        lambda: FailingS3Client("NoSuchKey"),
    )

    assert store.download_file("history.parquet", destination) == destination
    assert not destination.exists()
    assert store.exists("history.parquet") is False
    with pytest.raises(FileNotFoundError, match="Missing S3 artifact"):
        store.download_file("history.parquet", destination, required=True)


def test_optional_s3_download_propagates_access_failures(tmp_path, monkeypatch) -> None:
    store = ArtifactStore("s3://bucket/prefix")
    monkeypatch.setattr(
        "dkenergy_forecast.storage._s3_client",
        lambda: FailingS3Client("AccessDenied"),
    )

    with pytest.raises(FakeS3Error):
        store.download_file("history.parquet", tmp_path / "history.parquet")
    with pytest.raises(FakeS3Error):
        store.exists("history.parquet")
