"""Reproduce trace-level online scoring pagination with a migrated SQLite store."""

import argparse
import json
import os
import socket
import sqlite3
import tempfile
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch

os.environ.update(
    MLFLOW_ENABLE_ASYNC_TRACE_LOGGING="false",
    MLFLOW_ENABLE_WORKSPACES="false",
    MLFLOW_ONLINE_SCORING_MAX_WORKER_THREADS="2",
    MLFLOW_ONLINE_SCORING_DEFAULT_TRACE_COMPLETION_BUFFER_SECONDS="300",
    MLFLOW_CRYPTO_KEK_PASSPHRASE="fixture-local-passphrase",
    MLFLOW_ENABLE_TELEMETRY="false",
    LITELLM_LOCAL_MODEL_COST_MAP="true",
)

import mlflow
import mlflow.genai.scorers.job as job
from mlflow.entities import Feedback, GatewayEndpointModelConfig, GatewayModelLinkageType
from mlflow.genai.evaluation.entities import EvalResult
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers.online import trace_checkpointer, trace_processor
from mlflow.genai.scorers.online.constants import (
    EXCLUDE_EVAL_RUN_TRACES_FILTER,
    MAX_TRACES_PER_JOB,
)
from mlflow.server.constants import ARTIFACT_ROOT_ENV_VAR, BACKEND_STORE_URI_ENV_VAR
from mlflow.server.handlers import _get_tracking_store
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

NOW_MS = 1_790_000_000_000
FIRST_CUTOFF = NOW_MS - 300_000
TIED_TIME = FIRST_CUTOFF - 1_000
LATER_TIME = FIRST_CUTOFF + 5_000
SCORER_NAME = "fixture_trace_judge"


def deny_network(*args, **kwargs):
    raise AssertionError("Trace scoring validation forbids socket connections")


def register_judge(store, experiment_id):
    secret = store.create_gateway_secret(secret_name="fixture-key", secret_value={"api_key": "local"})
    model = store.create_gateway_model_definition(
        name="fixture-model", secret_id=secret.secret_id, provider="openai", model_name="gpt-4"
    )
    store.create_gateway_endpoint(
        name="fixture-endpoint",
        model_configs=[
            GatewayEndpointModelConfig(
                model_definition_id=model.model_definition_id,
                linkage_type=GatewayModelLinkageType.PRIMARY,
                weight=1.0,
            )
        ],
        created_by="fixture",
        usage_tracking=False,
        experiment_id=experiment_id,
    )
    judge = make_judge(
        name=SCORER_NAME,
        instructions="Evaluate {{ outputs }} for clarity",
        feedback_value_type=str,
        model="gateway:/fixture-endpoint",
    )
    assert not judge.is_session_level_scorer
    store.register_scorer(experiment_id, judge.name, json.dumps(judge.model_dump()))
    store.upsert_online_scoring_config(experiment_id, judge.name, 1.0)
    assert [scorer.name for scorer in store.get_active_online_scorers()] == [SCORER_NAME]


def log_trace(experiment_id, timestamp):
    start_ns = timestamp * 1_000_000
    span = mlflow.start_span_no_context(
        "fixture-answer",
        inputs={"question": "hello"},
        experiment_id=experiment_id,
        start_time_ns=start_ns,
    )
    span.end(outputs={"answer": "hello"}, end_time_ns=start_ns + 1_000_000)
    return span.trace_id


def inspect_full_window(store, experiment_id):
    """Independent control of actual store pagination without the 500-row loader cap."""
    trace_ids = []
    page_sizes = []
    next_page = None
    for _ in range(8):
        page, next_page = store.search_traces(
            experiment_ids=[experiment_id],
            filter_string=(
                f"trace.timestamp_ms >= {TIED_TIME} AND trace.timestamp_ms <= {LATER_TIME}"
                f" AND {EXCLUDE_EVAL_RUN_TRACES_FILTER}"
            ),
            max_results=100,
            order_by=["timestamp_ms ASC", "request_id ASC"],
            page_token=next_page,
        )
        page_sizes.append(len(page))
        trace_ids.extend((t.info if hasattr(t, "info") else t).trace_id for t in page)
        if not next_page:
            return trace_ids, page_sizes
    raise AssertionError("Unexpectedly unbounded pagination for the fixture")


def run_case(directory, tied_count):
    db_path = directory / "store.db"
    uri = f"sqlite:///{db_path}"
    os.environ[BACKEND_STORE_URI_ENV_VAR] = uri
    os.environ[ARTIFACT_ROOT_ENV_VAR] = (directory / "artifacts").as_uri()
    mlflow.set_tracking_uri(uri)
    store = _get_tracking_store()
    assert isinstance(store, SqlAlchemyStore)
    with sqlite3.connect(db_path) as connection:
        schema = connection.execute("SELECT version_num FROM alembic_version").fetchall()
    assert len(schema) == 1 and schema[0][0]
    experiment_id = store.create_experiment("online-trace-ties")
    register_judge(store, experiment_id)

    tied_ids = sorted(log_trace(experiment_id, TIED_TIME) for _ in range(tied_count))
    later_id = log_trace(experiment_id, LATER_TIME)
    expected = tied_ids + [later_id]
    assert len(set(expected)) == tied_count + 1
    full_window, page_sizes = inspect_full_window(store, experiment_id)
    assert full_window == expected

    labels = {trace_id: f"tied-{index:03d}" for index, trace_id in enumerate(tied_ids)}
    labels[later_id] = "later"
    checkpoint_manager = trace_checkpointer.OnlineTraceCheckpointManager(store, experiment_id)
    assert checkpoint_manager.get_checkpoint() is None
    lock = Lock()
    invoked = []
    dispatches = []
    ticks = []

    def score(*, eval_item, scorers, rate_limiter, max_retries):
        assert [scorer.name for scorer in scorers] == [SCORER_NAME]
        assert eval_item.trace.info.trace_id in labels
        assert len(eval_item.trace.data.spans) == 1
        trace_id = eval_item.trace.info.trace_id
        with lock:
            invoked.append(trace_id)
        return EvalResult(
            eval_item=eval_item,
            assessments=[Feedback(name=SCORER_NAME, value="valid", trace_id=trace_id)],
        )

    def dispatch(function, payload):
        assert function is job.run_online_trace_scorer_job
        assert payload["experiment_id"] == experiment_id
        assert [row["name"] for row in payload["online_scorers"]] == [SCORER_NAME]
        dispatches.append(payload["experiment_id"])
        function(**payload)

    def read_feedback_ids():
        ids = {}
        for trace_id in expected:
            matching = [
                assessment
                for assessment in store.get_trace_info(trace_id).assessments or []
                if assessment.name == SCORER_NAME
            ]
            if matching:
                assert len(matching) == 1 and matching[0].assessment_id
                assert matching[0].error is None and matching[0].value == "valid"
                ids[trace_id] = matching[0].assessment_id
        return ids

    clock = SimpleNamespace(time=lambda: NOW_MS / 1000)
    first_feedback = {}
    feedback = {}
    with (
        patch.object(trace_checkpointer, "time", clock),
        patch.object(trace_processor, "time", clock),
        patch.object(job, "submit_job", side_effect=dispatch),
        patch("mlflow.genai.evaluation.harness._compute_eval_scores", side_effect=score),
    ):
        for offset in (0, 60_000, 120_000):
            clock.time = lambda: (NOW_MS + offset) / 1000
            before = len(invoked)
            job.run_online_scoring_scheduler()
            feedback = read_feedback_ids()
            checkpoint = checkpoint_manager.get_checkpoint()
            assert checkpoint is not None
            if offset == 0:
                first_feedback = feedback
                expected_first = tied_ids[:MAX_TRACES_PER_JOB]
                assert set(first_feedback) == set(expected_first)
                assert Counter(invoked) == Counter(expected_first)
                assert asdict(checkpoint) == {
                    "timestamp_ms": TIED_TIME,
                    "trace_id": expected_first[-1],
                }
            assert all(feedback.get(trace_id) == aid for trace_id, aid in first_feedback.items())
            ticks.append(
                {
                    "offset_ms": offset,
                    "new_feedback": sorted(labels[trace_id] for trace_id in invoked[before:]),
                    "persisted_feedback_count": len(feedback),
                    "persisted_checkpoint": asdict(checkpoint),
                }
            )

    assert len(dispatches) == 3
    actual = Counter(invoked)
    wanted = Counter(expected)
    assert set(actual) == set(feedback)
    missing = sorted(labels[trace_id] for trace_id in (wanted - actual).elements())
    replayed = sorted(labels[trace_id] for trace_id in (actual - wanted).elements())
    result = {
        "tied_count": tied_count,
        "sqlite_version": sqlite3.sqlite_version,
        "alembic_revision": schema[0][0],
        "sqlalchemy_version": version("sqlalchemy"),
        "mlflow_source": mlflow.__file__,
        "loader_cap": MAX_TRACES_PER_JOB,
        "raw_store_page_sizes": page_sizes,
        "expected_feedback": len(expected),
        "persisted_feedback": len(feedback),
        "missing": {label: trace_id for trace_id, label in labels.items() if label in missing},
        "unexpected_or_replayed": replayed,
        "scheduler_dispatch": "synchronous seam calling the registered trace job",
        "provider": "deterministic _compute_eval_scores seam; sockets disabled",
        "ticks": ticks,
    }
    print("MLFLOW_SQLITE_TRACE_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    assert actual == wanted, f"Missing or repeated traces: {missing}, {replayed}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tied_count", type=int, choices=(499, 500, 501))
    args = parser.parse_args()
    with (
        tempfile.TemporaryDirectory(prefix="mlflow-trace-validation-") as directory,
        patch.object(socket.socket, "connect", deny_network),
        patch.object(socket.socket, "connect_ex", deny_network),
    ):
        run_case(Path(directory), args.tied_count)
