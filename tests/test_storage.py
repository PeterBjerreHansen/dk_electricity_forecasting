from __future__ import annotations

from pathlib import Path

from dkenergy_forecast.storage import ArtifactStore, join_uri, materialize_uri, parse_uri


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


def test_materialize_uri_returns_local_file_path(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text("{}", encoding="utf-8")

    assert materialize_uri(f"file://{path}", required=True) == path
