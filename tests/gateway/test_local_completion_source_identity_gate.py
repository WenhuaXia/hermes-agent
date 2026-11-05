"""FIX-2101: a local async-delegation completion's REBUILT source must pass the local-automation
admission gate, not be false-refused on object identity.

The completion path (``_inject_watch_notification`` -> ``admit_process_notice`` ->
``snapshot_local_automation``) rebuilds ``event.source`` via ``_build_process_event_source`` — a
fresh canonical ``local_source`` — so it is never the SAME Python object as the authority's
registered live source. The pre-FIX-2101 gate compared object identity (``live.source is not
event.source`` and ``authorize_source(event.source)``, which itself requires ``source is
original``) and refused EVERY local completion with ``permission_denied``. That refusal is
non-permanent (``PERMANENT_REFUSAL_REASONS`` = not_found/invalid_params), so the durable claim was
deferred (``delivery_attempts`` refunded to 0) and the watcher requeued it forever: the row sat
``delivery_state='pending', delivery_attempts=0`` and the parent session never saw the report
(production: deleg_ecdc8933 / deleg_1214c07c on local-a08b..., 5-day ``permission_denied`` loop).
The gate now compares the STABLE source identity (the same tuple ``authorize_source`` gates on) and
authorizes the LIVE source — a genuine session change still refuses. Mirrors FIX-2087's guard.
"""
from types import SimpleNamespace

import pytest


def _runner(tmp_path):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    return SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False)


async def _registered_local_session(tmp_path):
    """A local session whose restore has taken the register branch: canonical ``local_source`` live
    and registered with the LOCAL adapter, ready to be re-restored (same scaffolding as
    ``test_local_session_source_identity_guard``)."""
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import restore_local_session
    authority = await initialize_session_authority(_runner(tmp_path), profile_id='fixture', instance_id='first')
    actor = Principal('owner', 'fixture', frozenset({'session:create', 'session:read'}), 'socket')
    ref = create_local_session(authority, actor, {'request_id': 'r1', 'source': 'gui',
                                                  'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
    sid = ref.session_id
    authority.sessions.pop(sid, None)  # force the register branch on the first restore
    restore_local_session(authority, sid)
    return authority, sid


def _local_adapter(authority, sid):
    from gateway.session_local_recovery import local_adapter_map
    live = authority.sessions[sid]
    adapter = local_adapter_map(authority).get(live.source.platform)
    # The gate's third term resolves the adapter by the live source; bind it to the local adapter.
    authority.runner._adapter_for_source = lambda source: adapter
    return live, adapter


def _completion_event(fresh_source, entry):
    """The shape the completion path hands the gate: a FRESH source object (rebuilt), the same route
    metadata a real notification carries."""
    class _Evt:
        text = 'BACKGROUND REPORT'
        source = fresh_source
        metadata = {'gateway_session_key': entry.session_key,
                    'gateway_session_id': entry.session_id}
    return _Evt()


@pytest.mark.asyncio
async def test_rebuilt_completion_source_is_admitted(tmp_path):
    """The completion path's freshly-rebuilt source is the SAME session (stable identity intact);
    the gate must admit it, not refuse it. RED on the base object-identity gate (permission_denied),
    GREEN with the stable-identity comparison."""
    from gateway.session_automation import snapshot_local_automation
    from gateway.session_local_recovery import local_source
    authority, sid = await _registered_local_session(tmp_path)
    live, adapter = _local_adapter(authority, sid)
    entry = authority.runner.session_store.lookup_by_session_key(live.route)

    fresh = local_source(authority, live.source.chat_id, live.source.user_id)
    assert fresh is not live.source  # the whole bug: the completion path never reuses the live object

    payload, got_entry = snapshot_local_automation(authority, adapter,
                                                   _completion_event(fresh, entry), 'identity-1', entry)
    assert got_entry is entry
    assert payload['local_automation_v1']['owner'] == sid


@pytest.mark.asyncio
async def test_foreign_session_source_still_refuses(tmp_path):
    """Sanity: a source whose stable identity names a DIFFERENT session (different principal) must
    STILL be refused, so narrowing the gate to the stable identity did not over-broaden it."""
    from gateway.session_automation import snapshot_local_automation
    from gateway.session_local_recovery import local_source
    from hermes_state_runtime import RuntimeStoreError
    authority, sid = await _registered_local_session(tmp_path)
    live, adapter = _local_adapter(authority, sid)
    entry = authority.runner.session_store.lookup_by_session_key(live.route)

    foreign = local_source(authority, live.source.chat_id, 'a-different-principal')
    assert foreign is not live.source
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        snapshot_local_automation(authority, adapter,
                                  _completion_event(foreign, entry), 'identity-x', entry)
