"""Render research/to_human/progress-NNN.html from research-state.yaml."""

import html
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
state = yaml.safe_load((ROOT / "research-state.yaml").read_text(encoding="utf-8"))
out = ROOT / "to_human" / f"progress-{sys.argv[1]}.html"

# Shipped product states on the proxy metric (ms/frame, JSON-only, sample.mp4).
steps = [
    ("Baseline (CUDA EP FP16)", 52.31, "run_003 B"),
    ("+ TensorRT EP", 39.85, "run_003 T"),
    ("+ ball detection in parallel", 36.93, "run_004 P"),
]


def chart() -> str:
    w, h, pad = 760, 300, 60
    lo, hi = 30.0, 55.0
    y = lambda v: pad + (hi - v) / (hi - lo) * (h - 2 * pad)  # noqa: E731
    x = lambda i: pad + i * (w - 2 * pad) / (len(steps) - 1)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="ms per frame by shipped change">']
    for v in (30, 35, 40, 45, 50, 55):
        parts.append(f'<line x1="{pad}" x2="{w - pad}" y1="{y(v)}" y2="{y(v)}" class="grid"/>'
                     f'<text x="{pad - 8}" y="{y(v) + 4}" class="axis" text-anchor="end">{v}</text>')
    parts.append(f'<line x1="{pad}" x2="{w - pad}" y1="{y(steps[0][1])}" y2="{y(steps[0][1])}" class="base"/>')
    pts = " ".join(f"{x(i)},{y(v)}" for i, (_, v, _) in enumerate(steps))
    parts.append(f'<polyline points="{pts}" class="line"/>')
    for i, (label, v, _) in enumerate(steps):
        anchor = "start" if i == 0 else "end" if i == len(steps) - 1 else "middle"
        lx = x(i) - 6 if i == 0 else x(i) + 6 if i == len(steps) - 1 else x(i)
        parts.append(f'<circle cx="{x(i)}" cy="{y(v)}" r="5" class="dot"/>'
                     f'<text x="{x(i)}" y="{y(v) - 12}" text-anchor="middle" class="val">{v:.1f} ms</text>'
                     f'<text x="{lx}" y="{h - pad + 22}" text-anchor="{anchor}" class="axis">{html.escape(label)}</text>')
    parts.append(f'<text x="18" y="{h / 2}" class="axis" transform="rotate(-90 18 {h / 2})" '
                 'text-anchor="middle">ms / frame (lower is better)</text></svg>')
    return "".join(parts)


badge = {"supported": "ok", "refuted": "no", "inconclusive": "mid", "pending": "todo", "active": "todo"}
rows = "".join(
    f'<tr><td>{h["id"]}</td><td><span class="b {badge[h["status"]]}">{h["status"]}</span></td>'
    f'<td>{html.escape(h["statement"])}</td></tr>'
    for h in state["hypotheses"]
)

page = f"""<!doctype html><html lang="vi"><meta charset="utf-8">
<title>Pipeline throughput — progress {sys.argv[1]}</title>
<style>
body{{font:15px/1.55 system-ui,sans-serif;max-width:980px;margin:32px auto;padding:0 20px;color:#1d1d1f;background:#fff}}
h1{{font-size:24px;margin-bottom:4px}} h2{{font-size:18px;margin-top:32px;border-bottom:1px solid #ddd;padding-bottom:4px}}
.sub{{color:#666}} table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{border-bottom:1px solid #eee;padding:6px 8px;text-align:left;vertical-align:top}} th{{background:#f6f6f6}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}
.b{{padding:1px 8px;border-radius:10px;font-size:12px;white-space:nowrap}}
.ok{{background:#d8f3dc;color:#1b5e20}} .no{{background:#fde2e1;color:#8a1c1c}}
.mid{{background:#fff1c2;color:#7a5b00}} .todo{{background:#e8eaf6;color:#283593}}
.grid{{stroke:#e5e5e5}} .base{{stroke:#c62828;stroke-dasharray:6 4}} .line{{fill:none;stroke:#1565c0;stroke-width:3}}
.dot{{fill:#1565c0}} .axis{{font-size:12px;fill:#666}} .val{{font-size:13px;font-weight:600;fill:#1d1d1f}}
.kpi{{display:flex;gap:16px;margin:16px 0}} .kpi div{{flex:1;background:#f6f8fb;border-radius:10px;padding:12px 16px}}
.kpi b{{display:block;font-size:24px}}
</style>
<h1>Tối ưu thông lượng pipeline phân tích bóng đá</h1>
<div class="sub">RTX 3050 Ti Laptop · data/sample.mp4 (1505 frame, 1280×720) · cấu hình mặc định, JSON-only · đo xen kẽ để loại nhiễu nhiệt</div>

<div class="kpi"><div><b>52.3 → 36.9 ms</b>ms/frame đã ship</div><div><b>19.1 → 27.1 fps</b>−29.4% thời gian</div>
<div><b>Output giữ nguyên</b>recall cầu thủ 0.997; bóng &amp; cú sút trùng khớp</div></div>

<h2>Quỹ đạo tối ưu</h2>
{chart()}

<h2>Phát hiện chính</h2>
<ol>
<li><b>Lúc đầu GPU là nút thắt.</b> Forward thuần: cầu thủ 14.9 ms, bóng 28.6 ms. Copy dữ liệu không đáng kể (IO binding chỉ tiết kiệm ≤ 1.7 ms).</li>
<li><b>TensorRT FP16 nhanh 1.7–1.8× mỗi model</b> với output gần như trùng CUDA FP16 (IoU ≥ 0.994). End-to-end −23.8%. Ultralytics luôn mở ONNX bằng CUDA EP, nên cần thay session sau lần gọi thật đầu tiên.</li>
<li><b>Chạy model bóng song song với tracking cầu thủ</b> giấu được ~35% chi phí của nó (−3.1 ms), output trùng bit.</li>
<li><b>Sau TensorRT, nút thắt chuyển sang CPU:</b> OC-SORT ~6.5–9 ms, letterbox trên CPU ~3 ms, trong khi GPU chỉ còn 8.8 ms.</li>
<li><b>Hai kết quả âm có giá trị:</b>
  <ul>
  <li>ByteTrack nhanh hơn 4 ms nhưng <b>tăng 35% đổi ID</b> trên ground truth SportsMOT. Timeline gán quyền giữ bóng theo ID, nên giữ OC-SORT.</li>
  <li>Bật lại đa luồng OpenCV gây tranh chấp CPU: chậm hơn và thất thường. CPU đã bão hoà, chỉ còn cách <i>bớt việc</i>, không thể song song hoá thêm.</li>
  </ul></li>
</ol>

<h2>Bản đồ giả thuyết</h2>
<table><tr><th>ID</th><th>Trạng thái</th><th>Phát biểu (đã khoá trước khi chạy)</th></tr>{rows}</table>

<h2>Bước tiếp theo</h2>
<ul>
<li><b>H7b</b>: letterbox trên GPU (upload uint8, resize+pad bằng torch). Kỳ vọng −3 ms CPU mỗi lần gọi model cầu thủ, có guard parity.</li>
<li><b>H9</b>: model cầu thủ ở imgsz 768 thay vì 960. Đánh đổi độ chính xác, đo IDF1 trên ground truth SportsMOT; ngưỡng ≤ 1 điểm.</li>
<li>OC-SORT vectorised với output giống hệt: lợi lớn nhất còn lại (~5 ms) nhưng là một dự án code riêng.</li>
</ul>
<p class="sub">Chi tiết từng thí nghiệm: <code>research/experiments/*/protocol.md</code> (khoá trong git trước khi chạy) và <code>analysis.md</code>.</p>
</html>"""

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(page, encoding="utf-8")
print(out)
