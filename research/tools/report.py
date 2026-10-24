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
    ("Baseline", 52.31, "run_003 B"),
    ("+ TensorRT", 39.85, "run_003 T"),
    ("+ bóng song song", 36.93, "run_004 P"),
    ("+ letterbox GPU", 32.45, "run_008 G"),
    ("+ OC-SORT batch", 27.55, "run_012 V"),
]


def chart() -> str:
    w, h, pad = 760, 300, 60
    lo, hi = 25.0, 55.0
    y = lambda v: pad + (hi - v) / (hi - lo) * (h - 2 * pad)  # noqa: E731
    x = lambda i: pad + i * (w - 2 * pad) / (len(steps) - 1)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="ms per frame by shipped change">']
    for v in (25, 30, 35, 40, 45, 50, 55):
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

<div class="kpi"><div><b>52.3 → 27.6 ms</b>ms/frame đã ship</div><div><b>19.1 → 36.3 fps</b>−47% thời gian</div>
<div><b>Output an toàn</b>TensorRT/letterbox: lệch ở mức nhiễu FP16 · song song bóng &amp; OC-SORT: trùng từng byte</div></div>

<h2>Quỹ đạo tối ưu</h2>
{chart()}
<p class="sub">Mỗi điểm là trung bình của biến thể trong lần đo xen kẽ riêng của nó; baseline giữa các phiên lệch ≤ 0.9 ms.</p>

<h2>Vòng 2: ba ý tưởng</h2>
<table><tr><th>Ý tưởng</th><th>Kết quả</th><th>Quyết định</th></tr>
<tr><td>Letterbox trên GPU (H7b)</td><td>37.81 → 32.45 ms (−5.4 ms, dự đoán ≥ 2 ms). Input bóng trùng từng bit; input cầu thủ lệch ±1 mức xám; recall 0.996</td><td><span class="b ok">ship</span></td></tr>
<tr><td>Input model cầu thủ nhỏ hơn (H9)</td><td>768×768: IDF1 −5.9 · 544×960: đủ guard nhưng chỉ −1.38 ms, và đổi 5.5% box</td><td><span class="b no">không ship</span></td></tr>
<tr><td>OC-SORT batch (H6c → H6d)</td><td>Lần 1 thiếu bước clip box ở mép ảnh → khác 56 frame. Lần 2: 32.09 → 27.55 ms (−4.54 ms), frames.json trùng từng byte</td><td><span class="b ok">ship</span></td></tr>
</table>

<h2>Phát hiện chính</h2>
<ol>
<li><b>Bớt việc CPU có lợi hơn chính thời gian của nó.</b> Letterbox ~3 ms nhưng tiết kiệm 5.4 ms; tracker ~3 ms nhưng tiết kiệm 4.5 ms, vì luồng chính và luồng bóng tranh nhau CPU. Ngược lại, thêm luồng OpenCV (H8) làm chậm hơn.</li>
<li><b>Cùng tỉ lệ vẫn không cùng kết quả.</b> Bỏ padding (544×960) giữ nguyên tỉ lệ 0.75, nhưng detector bớt bắt khán giả phía trên ảnh và sót thêm một ít cầu thủ giữa sân: hình học input cũng là thay đổi độ chính xác.</li>
<li><b>MOTA đánh lừa ở đây, IDF1 mới quyết định.</b> Detector COCO bắt cả người ngoài sân không có nhãn; giảm input làm MOTA tăng nhưng IDF1 giảm 3–10 điểm vì mất cầu thủ ở xa.</li>
<li><b>"Trùng output" phải kiểm ở ranh giới sản phẩm.</b> Port OC-SORT khớp chính xác ở mức tracker ngay từ đầu, nhưng sản phẩm còn một bước clip của Ultralytics phía sau; chỉ so sánh frames.json mới phát hiện ra.</li>
</ol>

<h2>Bản đồ giả thuyết</h2>
<table><tr><th>ID</th><th>Trạng thái</th><th>Phát biểu (đã khoá trước khi chạy)</th></tr>{rows}</table>

<h2>Còn lại</h2>
<ul>
<li><b>Padding của model bóng</b>: input 1280×1280 cho frame 1280×720, 44% là padding, trên forward 13 ms. Cần checkpoint huấn luyện để export lại, và phải kiểm lại recall bóng (bài học H9b).</li>
<li><b>Hai detector trên một GPU 4 GB</b>: <code>player_track</code> chạy riêng 15 ms nhưng ~22 ms khi chạy cạnh model bóng; cần thử cách sắp lịch hai model.</li>
<li><b>Xuất video (H5)</b>: bật output video thêm ~20 ms/frame (render 2.7 + encode 7.2 ms đo được, phần còn lại là vẽ và tranh chấp); nên đưa encode ra khỏi luồng chính nếu cần xuất video.</li>
</ul>
<p class="sub">Chi tiết từng thí nghiệm: <code>research/experiments/*/protocol.md</code> (khoá trong git trước khi chạy) và <code>analysis.md</code>.</p>
</html>"""

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(page, encoding="utf-8")
print(out)
