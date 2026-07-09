from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse


@dataclass(frozen=True)
class ParsedUri:
    scheme: str
    path: str
    bucket: str | None = None
    key: str = ""

    @property
    def is_s3(self) -> bool:
        return self.scheme == "s3"

    @property
    def is_local(self) -> bool:
        return self.scheme in {"", "file"}


class StorageError(RuntimeError):
    """Raised when artifact storage cannot read or write a requested object."""


def parse_uri(value: str | Path) -> ParsedUri:
    text = str(value)
    parsed = urlparse(text)
    if parsed.scheme == "s3":
        if not parsed.netloc:
            raise ValueError(f"S3 URI must include a bucket: {text!r}")
        return ParsedUri(
            scheme="s3",
            bucket=parsed.netloc,
            path=text,
            key=parsed.path.lstrip("/"),
        )
    if parsed.scheme == "file":
        return ParsedUri(scheme="file", path=unquote(parsed.path))
    if parsed.scheme:
        raise ValueError(f"Unsupported artifact URI scheme: {parsed.scheme!r}")
    return ParsedUri(scheme="", path=text)


def join_uri(base_uri: str | Path, *parts: str) -> str:
    parsed = parse_uri(base_uri)
    clean_parts = [part.strip("/") for part in parts if part and part.strip("/")]
    if parsed.is_s3:
        key = _join_key(parsed.key, *clean_parts)
        return f"s3://{parsed.bucket}/{key}" if key else f"s3://{parsed.bucket}"
    return str(Path(parsed.path, *clean_parts))


def materialize_uri(
    uri: str | Path,
    *,
    cache_dir: str | Path | None = None,
    required: bool = False,
) -> Path:
    """Return a local path for a local, file://, or s3:// object URI."""

    parsed = parse_uri(uri)
    if parsed.is_local:
        path = Path(parsed.path).expanduser()
        if required and not path.exists():
            raise FileNotFoundError(f"Missing local artifact: {path}")
        return path

    cache_root = Path(cache_dir or "/tmp/dkenergy-artifact-cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    filename = Path(parsed.key).name or "artifact"
    digest = hashlib.sha256(str(uri).encode("utf-8")).hexdigest()[:16]
    destination = cache_root / f"{digest}-{filename}"
    try:
        _s3_client().download_file(parsed.bucket, parsed.key, str(destination))
    except Exception as exc:  # pragma: no cover - exercised with real AWS only
        if required:
            raise FileNotFoundError(f"Missing S3 artifact: {uri}") from exc
        return destination
    return destination


def resource_exists(uri: str | Path) -> bool:
    parsed = parse_uri(uri)
    if parsed.is_local:
        return Path(parsed.path).expanduser().exists()
    try:
        _s3_client().head_object(Bucket=parsed.bucket, Key=parsed.key)
        return True
    except Exception:  # pragma: no cover - exercised with real AWS only
        return False


class ArtifactStore:
    def __init__(self, root_uri: str | Path) -> None:
        self.root_uri = str(root_uri)
        self.root = parse_uri(root_uri)

    def uri_for(self, key: str = "") -> str:
        return join_uri(self.root_uri, key)

    def exists(self, key: str) -> bool:
        if self.root.is_local:
            return (Path(self.root.path).expanduser() / key).exists()
        remote_key = self._remote_key(key)
        try:
            if remote_key.endswith("/"):
                return bool(self._list_s3_keys(remote_key))
            _s3_client().head_object(Bucket=self.root.bucket, Key=remote_key)
            return True
        except Exception:  # pragma: no cover - exercised with real AWS only
            return False

    def download_prefix(
        self,
        prefix: str,
        destination: str | Path,
        *,
        required: bool = False,
    ) -> list[Path]:
        destination_path = Path(destination)
        destination_path.mkdir(parents=True, exist_ok=True)
        if self.root.is_local:
            source = Path(self.root.path).expanduser() / prefix
            if not source.exists():
                if required:
                    raise FileNotFoundError(f"Missing artifact store prefix: {source}")
                return []
            if source.is_file():
                output = destination_path / source.name
                shutil.copy2(source, output)
                return [output]
            return _copy_tree_contents(source, destination_path)

        remote_prefix = self._remote_key(prefix)
        keys = self._list_s3_keys(remote_prefix)
        if required and not keys:
            raise FileNotFoundError(f"Missing S3 artifact prefix: {self.uri_for(prefix)}")
        written: list[Path] = []
        for key in keys:
            relative = Path(key.removeprefix(remote_prefix).lstrip("/"))
            if not str(relative):
                continue
            output = destination_path / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            _s3_client().download_file(self.root.bucket, key, str(output))
            written.append(output)
        return written

    def upload_prefix(self, source: str | Path, prefix: str) -> list[str]:
        source_path = Path(source)
        if not source_path.exists():
            return []
        uploaded: list[str] = []
        for path in _iter_files(source_path):
            relative = path.relative_to(source_path).as_posix()
            key = _join_key(prefix, relative)
            self.upload_file(path, key)
            uploaded.append(key)
        return uploaded

    def download_file(
        self,
        key: str,
        destination: str | Path,
        *,
        required: bool = False,
    ) -> Path:
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if self.root.is_local:
            source = Path(self.root.path).expanduser() / key
            if not source.exists():
                if required:
                    raise FileNotFoundError(f"Missing artifact store object: {source}")
                return destination_path
            shutil.copy2(source, destination_path)
            return destination_path

        remote_key = self._remote_key(key)
        try:
            _s3_client().download_file(self.root.bucket, remote_key, str(destination_path))
        except Exception as exc:  # pragma: no cover - exercised with real AWS only
            if required:
                raise FileNotFoundError(f"Missing S3 artifact: {self.uri_for(key)}") from exc
        return destination_path

    def upload_file(self, source: str | Path, key: str) -> str:
        source_path = Path(source)
        if self.root.is_local:
            destination = Path(self.root.path).expanduser() / key
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            return str(destination)
        remote_key = self._remote_key(key)
        _s3_client().upload_file(str(source_path), self.root.bucket, remote_key)
        return f"s3://{self.root.bucket}/{remote_key}"

    def _remote_key(self, key: str) -> str:
        return _join_key(self.root.key, key)

    def _list_s3_keys(self, prefix: str) -> list[str]:
        client = _s3_client()
        keys: list[str] = []
        token: str | None = None
        while True:  # pragma: no cover - exercised with real AWS only
            kwargs = {"Bucket": self.root.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            response = client.list_objects_v2(**kwargs)
            keys.extend(
                item["Key"]
                for item in response.get("Contents", [])
                if not item["Key"].endswith("/")
            )
            if not response.get("IsTruncated"):
                return keys
            token = response.get("NextContinuationToken")


def _copy_tree_contents(source: Path, destination: Path) -> list[Path]:
    written: list[Path] = []
    for path in _iter_files(source):
        relative = path.relative_to(source)
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, output)
        written.append(output)
    return written


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for child in sorted(path.rglob("*")):
        if child.is_file():
            yield child


def _join_key(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part and part.strip("/"))


def _s3_client():
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "S3 artifact storage requires boto3. Install with `pip install -e \".[aws]\"`."
        ) from exc
    return boto3.client("s3")
