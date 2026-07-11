from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import boto3


S3 = boto3.client("s3")
SNS = boto3.client("sns")


def handler(event: dict[str, object], context: object) -> dict[str, object]:
    del event, context
    now = datetime.now(timezone.utc)
    problems: list[str] = []

    try:
        pointer, last_modified = _read_json(
            os.environ["ARTIFACT_BUCKET"],
            os.environ["PUBLICATION_MARKER_KEY"],
        )
        _validate_pointer(pointer, last_modified, now, problems)
    except Exception as exc:  # The alert is the recovery path for a missing/broken pointer.
        problems.append(f"could not read the publication pointer: {exc}")

    if not problems:
        return {"ok": True}

    message = (
        "The Danish electricity day-ahead forecast did not pass its publication "
        "deadline check:\n- " + "\n- ".join(problems)
    )
    print(message)
    SNS.publish(
        TopicArn=os.environ["ALERT_TOPIC_ARN"],
        Subject="DK electricity forecast deadline missed",
        Message=message,
    )
    return {"ok": False, "problems": problems}


def _validate_pointer(
    pointer: dict[str, object],
    last_modified: datetime,
    now: datetime,
    problems: list[str],
) -> None:
    expected_date = (
        now.astimezone(ZoneInfo(os.environ["SCHEDULE_TIMEZONE"])).date()
        + timedelta(days=int(os.environ["DELIVERY_DATE_OFFSET_DAYS"]))
    ).isoformat()
    if pointer.get("status") != "completed":
        problems.append(f"latest.json status is {pointer.get('status')!r}, not 'completed'")
    if pointer.get("delivery_date_local") != expected_date:
        problems.append(
            "latest.json delivery_date_local is "
            f"{pointer.get('delivery_date_local')!r}; expected {expected_date!r}"
        )

    maximum_age = timedelta(minutes=int(os.environ["MARKER_MAX_AGE_MINUTES"]))
    if now - last_modified > maximum_age:
        problems.append(
            f"latest.json was last modified at {last_modified.isoformat()}, "
            f"more than {maximum_age} ago"
        )

    committed_at = _parse_timestamp(pointer.get("committed_at_utc"), "committed_at_utc", problems)
    deadline = _parse_timestamp(pointer.get("decision_deadline_utc"), "decision_deadline_utc", problems)
    if committed_at is not None and now - committed_at > maximum_age:
        problems.append(
            f"latest.json committed_at_utc is {committed_at.isoformat()}, "
            f"more than {maximum_age} ago"
        )
    if committed_at is not None and deadline is not None and committed_at > deadline:
        problems.append(
            f"forecast committed at {committed_at.isoformat()}, after its "
            f"{deadline.isoformat()} decision deadline"
        )

    completion_key = pointer.get("completion_key")
    if not isinstance(completion_key, str) or not completion_key.strip("/"):
        problems.append("latest.json has no valid completion_key")
        return

    try:
        completion, _ = _read_json(
            os.environ["ARTIFACT_BUCKET"],
            _artifact_key(completion_key),
        )
    except Exception as exc:
        problems.append(f"referenced completion receipt is unavailable: {exc}")
        return

    if completion.get("run_id") not in {None, pointer.get("run_id")}:
        problems.append("completion receipt run_id does not match latest.json")
    if completion.get("status") not in {None, "completed"}:
        problems.append(f"completion receipt status is {completion.get('status')!r}")


def _read_json(bucket: str, key: str) -> tuple[dict[str, object], datetime]:
    response = S3.get_object(Bucket=bucket, Key=key)
    body = json.loads(response["Body"].read())
    if not isinstance(body, dict):
        raise ValueError(f"s3://{bucket}/{key} is not a JSON object")
    return body, response["LastModified"].astimezone(timezone.utc)


def _artifact_key(relative_key: str) -> str:
    key = relative_key.strip("/")
    prefix = os.environ.get("ARTIFACT_PREFIX", "").strip("/")
    if not prefix or key == prefix or key.startswith(f"{prefix}/"):
        return key
    return f"{prefix}/{key}"


def _parse_timestamp(
    value: object,
    field: str,
    problems: list[str],
) -> datetime | None:
    if not isinstance(value, str):
        problems.append(f"latest.json has no valid {field}")
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        problems.append(f"latest.json {field} is not an ISO-8601 timestamp")
        return None
    if parsed.tzinfo is None:
        problems.append(f"latest.json {field} has no timezone")
        return None
    return parsed.astimezone(timezone.utc)
