"""Exercise actual MLflow scheduling, migrated SQLite, trace export and assessments."""

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
    MLFLOW_CRYPTO_KEK_PASSPHRASE="fixture-local-passphrase",
    MLFLOW_ENABLE_TELEMETRY="false",
    LITELLM_LOCAL_MODEL_COST_MAP="true",
)

import mlflow
from mlflow.entities import Feedback, GatewayEndpointModelConfig, GatewayModelLinkageType
from mlflow.genai.evaluation.entities import EvalResult
from mlflow.genai.judges import make_judge
import mlflow.genai.scorers.job as job
from mlflow.genai.scorers.online import session_checkpointer
from mlflow.genai.scorers.online.constants import MAX_SESSIONS_PER_JOB
from mlflow.server.constants import ARTIFACT_ROOT_ENV_VAR, BACKEND_STORE_URI_ENV_VAR
from mlflow.server.handlers import _get_tracking_store
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.constant import AssessmentMetadataKey, TraceMetadataKey

BUFFER_SECONDS = 300
NOW_MS = 1_790_000_000_000
FIRST_CUTOFF = NOW_MS - BUFFER_SECONDS * 1000
SCORER_NAME = "conversation_judge"


def deny_network(*args, **kwargs):
    raise AssertionError("Scoring validation forbids network connections")


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
        instructions="Evaluate {{ conversation }} for quality",
        feedback_value_type=str,
        model="gateway:/fixture-endpoint",
    )
    store.register_scorer(experiment_id, judge.name, json.dumps(judge.model_dump()))
    store.upsert_online_scoring_config(experiment_id, judge.name, 1.0)
    scorers = store.get_active_online_scorers()
    assert [scorer.name for scorer in scorers] == [SCORER_NAME]


def log_session(experiment_id, session_id, timestamp):
    start_ns = timestamp * 1_000_000
    span = mlflow.start_span_no_context(
        "fixture-chat",
        inputs={"messages": [{"role": "user", "content": "hello"}]},
        metadata={TraceMetadataKey.TRACE_SESSION: session_id},
        experiment_id=experiment_id,
        start_time_ns=start_ns,
    )
    span.end(outputs={"content": "hello"}, end_time_ns=start_ns + 1_000_000)
    return span.trace_id


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
    experiment_id = store.create_experiment("online-session-ties")
    register_judge(store, experiment_id)
    checkpoint_manager = session_checkpointer.OnlineSessionCheckpointManager(store, experiment_id)
    tied_sessions = [f"tied-{index:03d}" for index in range(tied_count)]
    trace_ids = {
        session: log_session(experiment_id, session, FIRST_CUTOFF - 1_000)
        for session in tied_sessions
    }
    trace_ids["later"] = log_session(experiment_id, "later", FIRST_CUTOFF + 5_000)

    expected_first = tied_sessions[:MAX_SESSIONS_PER_JOB]
    first_page = store.find_completed_sessions(
        experiment_id=experiment_id,
        min_last_trace_timestamp_ms=FIRST_CUTOFF - 20_000,
        max_last_trace_timestamp_ms=FIRST_CUTOFF,
        max_results=MAX_SESSIONS_PER_JOB,
    )
    assert [session.session_id for session in first_page] == expected_first
    assert all(session.last_trace_timestamp_ms == FIRST_CUTOFF - 1_000 for session in first_page)

    def score(*, session_id, session_items, multi_turn_scorers):
        assert [scorer.name for scorer in multi_turn_scorers] == [SCORER_NAME]
        assert len(session_items) == 1
        item = session_items[0]
        assert item.trace.info.trace_id == trace_ids[session_id]
        assert len(item.trace.data.spans) == 1
        with lock:
            scored.append(session_id)
        return EvalResult(
            eval_item=item,
            assessments=[Feedback(name=SCORER_NAME, value="valid", trace_id=item.trace.info.trace_id)],
        )

    def dispatch(function, payload):
        assert function is job.run_online_session_scorer_job
        assert payload["experiment_id"] == experiment_id
        dispatches.append(payload["experiment_id"])
        function(**payload)

    def read_assessment_ids():
        found = {}
        for session_id, trace_id in trace_ids.items():
            assessments = store.get_trace_info(trace_id).assessments or []
            matching = [
                assessment.assessment_id
                for assessment in assessments
                if assessment.name == SCORER_NAME
                and (assessment.metadata or {}).get(AssessmentMetadataKey.ONLINE_SCORING_SESSION_ID)
                == session_id
            ]
            if matching:
                assert len(matching) == 1 and matching[0], (session_id, matching)
                found[session_id] = matching[0]
        return found

    lock = Lock()
    scored = []
    dispatches = []
    observations = []
    assessment_history = []
    clock = SimpleNamespace(time=lambda: NOW_MS / 1000)
    with (
        patch.object(session_checkpointer, "time", clock),
        patch.dict(
            os.environ,
            {"MLFLOW_ONLINE_SCORING_DEFAULT_SESSION_COMPLETION_BUFFER_SECONDS": str(BUFFER_SECONDS)},
        ),
        patch.object(job, "submit_job", side_effect=dispatch),
        patch("mlflow.genai.evaluation.session_utils.evaluate_session_level_scorers", side_effect=score),
    ):
        for offset in (0, 60_000, 120_000):
            clock.time = lambda: (NOW_MS + offset) / 1000
            before = len(scored)
            job.run_online_scoring_scheduler()
            assessment_ids = read_assessment_ids()
            assessment_history.append(assessment_ids)
            observations.append(
                {
                    "offset_ms": offset,
                    "newly_scored": sorted(scored[before:]),
                    "persisted_checkpoint": asdict(checkpoint_manager.get_checkpoint()),
                    "persisted_assessments": assessment_ids,
                }
            )
    assert len(dispatches) == 3
    assert observations[0]["newly_scored"] == expected_first
    assert set(assessment_history[0]) == set(expected_first)
    assert all(assessment_history[-1].get(session) == aid for session, aid in assessment_history[0].items())

    expected = Counter(trace_ids.keys())
    actual = Counter(scored)
    assert set(actual) == set(assessment_history[-1])
    result = {
        "case": tied_count,
        "sqlite_version": sqlite3.sqlite_version,
        "alembic_revision": schema[0][0],
        "sqlalchemy_version": version("sqlalchemy"),
        "huey_version": version("huey"),
        "mlflow_source": mlflow.__file__,
        "actual_sql_page_limit": MAX_SESSIONS_PER_JOB,
        "expected_count": sum(expected.values()),
        "observed_count": sum(actual.values()),
        "missing": sorted((expected - actual).elements()),
        "unexpected_or_replayed": sorted((actual - expected).elements()),
        "scheduler_dispatch": "synchronous seam calling the real registered session job",
        "provider": "evaluation-only seam; sockets disabled",
        "ticks": observations,
    }
    print("MLFLOW_SQLITE_SESSION_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    assert actual == expected, f"Missing or repeated sessions: {result['missing']}, {result['unexpected_or_replayed']}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tied_count", type=int, choices=(99, 100, 101))
    args = parser.parse_args()
    with (
        tempfile.TemporaryDirectory(prefix="mlflow-session-validation-") as directory,
        patch.object(socket.socket, "connect", deny_network),
        patch.object(socket.socket, "connect_ex", deny_network),
    ):
        run_case(Path(directory), args.tied_count)
