# Streaming Protocol Research — MJPEG vs SSE vs WebSocket

> **Question:** For streaming annotated football frames to a browser, should we use MJPEG (multipart/x-mixed-replace), SSE, or WebSocket?
>
> **Date:** 2025-08-09
>
> **Method:** Primary sources — WHATWG HTML spec, MDN, Starlette docs, W3C.

---

## 1. MJPEG (`multipart/x-mixed-replace`)

**What it is:** Server responds to GET with `Content-Type: multipart/x-mixed-replace; boundary=...`, streaming JPEG frames as separate parts. Browser renders via plain `<img src>` tag.

**Key facts from sources:**

- **Not formally standardized.** The MIME type is registered in [WHATWG HTML spec IANA considerations](https://html.spec.whatwg.org/multipage/iana.html) but was never standardized as an HTTP streaming mechanism — it's a Netscape-era feature ([Stack Overflow MJPEG spec discussion](https://stackoverflow.com/questions/47729941/mjpeg-over-http-specification)).
- **Browser issues with chunked encoding.** Chrome and Firefox have trouble combining chunked transfer-encoding with `multipart/x-mixed-replace` ([libmicrohttpd mailing list](https://lists.nongnu.org/archive/html/libmicrohttpd/2022-02/msg00010.html)).
- **Fetch API can't parse frames.** `response.body` ReadableStream doesn't expose the multipart boundary, making programmatic parsing hard ([WHATWG fetch issue #1021](https://github.com/whatwg/fetch/issues/1021)).
- **Starlette supports it.** `StreamingResponse` with `media_type='multipart/x-mixed-replace; boundary=frame'` is the documented way to serve MJPEG in FastAPI/Starlette ([Starlette Responses docs](https://www.starlette.io/responses/)).

**Pros:** `<img src>` works with zero JS; binary native; connection close = natural disconnect signal.

**Cons:** Non-standard; browser chunked-encoding quirks; no bidirectional control; hard to parse programmatically.

---

## 2. SSE (`text/event-stream`)

**What it is:** Standardized one-way server-push over HTTP. `Content-Type: text/event-stream`, messages as `data: ...\n\n`. Browser consumes via `EventSource` API.

**Key facts from sources:**

- **Fully standardized.** Defined in [WHATWG HTML spec server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html), documented in [MDN Using Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events).
- **Text-only.** Binary must be base64-encoded → ~33% overhead per frame. No native binary support.
- **Auto-reconnect built-in.** Browser reconnects automatically, `id` field for resume.
- **6-connection limit** per browser+domain under HTTP/1.1 (HTTP/2 removes it, default 100 streams) — [MDN warning](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events).

**Pros:** Standardized, auto-reconnect, simple server implementation.

**Cons:** Base64 overhead for images; not designed for video (text protocol); 6-conn limit on HTTP/1.1.

---

## 3. WebSocket

**What it is:** Full-duplex protocol over HTTP upgrade. Binary frames natively supported (`ws.send_bytes`). Requires JS client.

**Key facts from sources:**

- **Bidirectional.** Best for two-way interaction ([GetStream WebSocket vs SSE comparison](https://getstream.io/blog/websocket-sse/)).
- **Native binary.** No base64 overhead — send JPEG bytes directly.
- **Requires JS.** No `<img>` equivalent; client must decode and render via `<img>`/canvas.

**Pros:** Binary native, bidirectional (client can send pause/stop/seek), clean disconnect signal.

**Cons:** More complex; no auto-reconnect (must implement); JS required.

---

## Recommendation

| Criterion | MJPEG | SSE | WebSocket |
|-----------|-------|-----|-----------|
| Binary video frames | ✅ native | ❌ base64 overhead | ✅ native |
| Zero-JS `<img>` | ✅ | ❌ | ❌ |
| Standardized | ❌ | ✅ | ✅ |
| Bidirectional control | ❌ | ❌ | ✅ |
| Auto-reconnect | ❌ | ✅ | ❌ |

**For streaming annotated frames (binary JPEG):**

- **MJPEG** is the pragmatic choice — `<img src>` works, binary native, disconnect = close. Accept the non-standard caveat (it works in all modern browsers for this use case despite not being formalized).
- **WebSocket** if we later need client → server control (pause, seek, change thresholds). Currently we don't.
- **SSE** is wrong for video (text-only + base64 overhead).

**Decision:** MJPEG for the frame stream + a separate JSON endpoint for progress/stats.
