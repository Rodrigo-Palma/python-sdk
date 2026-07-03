"""Test for issue #3060 - Duplicate in-flight request id cross-wires two POSTs.

In the stateful StreamableHTTP server transport the per-request routing table
`_request_streams` is keyed by JSON-RPC request id alone. A second concurrent POST
that reuses an id already in flight silently OVERWRITES the earlier request's slot,
so the earlier caller's response is delivered to the wrong request (a cross-request
data leak) while the earlier request hangs forever.

The JSON-RPC id must be unique within a session, so a duplicate in-flight id is a
protocol violation that must be rejected loudly (HTTP 409 / JSON-RPC -32600) instead
of mis-routing. This test drives the real POST handler with an id that is already
present in `_request_streams` and asserts the second POST is rejected without
overwriting the first request's slot.
"""

import json

import anyio
import pytest
from mcp_types import INVALID_REQUEST
from starlette.types import Message, Scope

from mcp.server.streamable_http import EventMessage, StreamableHTTPServerTransport
from mcp.server.transport_security import TransportSecuritySettings
from mcp.shared._context_streams import create_context_streams
from mcp.shared.message import SessionMessage

DUPLICATE_REQUEST_ID = "in-flight-id"


def _make_post_receive(body: bytes):
    """ASGI receive that yields the request body once, then parks (open connection)."""
    delivered = False
    parked = anyio.Event()  # never set: keeps the connection open after the body

    async def receive() -> Message:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await parked.wait()
        return {"type": "http.disconnect"}

    return receive


@pytest.mark.anyio
async def test_duplicate_in_flight_request_id_is_rejected() -> None:
    """A POST reusing an in-flight request id is rejected with 409 / -32600.

    The earlier request's routing slot must survive untouched instead of being
    overwritten by the duplicate.
    """
    transport = StreamableHTTPServerTransport(
        mcp_session_id=None,
        is_json_response_enabled=True,
        security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    read_stream_writer, read_stream = create_context_streams[SessionMessage | Exception](0)
    transport._read_stream_writer = read_stream_writer  # pyright: ignore[reportPrivateUsage]

    # Simulate an earlier request already in flight under the same id.
    in_flight_send, in_flight_receive = anyio.create_memory_object_stream[EventMessage](1)
    in_flight_slot = (in_flight_send, in_flight_receive)
    transport._request_streams[DUPLICATE_REQUEST_ID] = in_flight_slot  # pyright: ignore[reportPrivateUsage]

    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": DUPLICATE_REQUEST_ID,
            "params": {},
        }
    ).encode()
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"accept", b"application/json, text/event-stream"),
        ],
    }

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    async with read_stream_writer, read_stream, in_flight_send, in_flight_receive:
        async with anyio.create_task_group() as tg:
            tg.start_soon(transport.handle_request, scope, _make_post_receive(body), send)
            # Bound the wait: with the guard the handler answers immediately; without it the
            # handler overwrites the slot and blocks forever waiting for a response.
            with anyio.move_on_after(2.0):
                while not any(m["type"] == "http.response.start" for m in sent):
                    await anyio.sleep(0.01)
            tg.cancel_scope.cancel()

    # The duplicate id must be rejected loudly, not silently mis-routed.
    start = next((m for m in sent if m["type"] == "http.response.start"), None)
    assert start is not None, "duplicate in-flight request id was not rejected (no response sent)"
    assert start["status"] == 409

    body_chunks = b"".join(m["body"] for m in sent if m["type"] == "http.response.body")
    payload = json.loads(body_chunks)
    assert payload["error"]["code"] == INVALID_REQUEST  # -32600

    # The earlier request's slot must be untouched, not overwritten by the duplicate.
    assert (
        transport._request_streams[DUPLICATE_REQUEST_ID] is in_flight_slot  # pyright: ignore[reportPrivateUsage]
    ), "duplicate POST overwrote the in-flight request's routing slot"
