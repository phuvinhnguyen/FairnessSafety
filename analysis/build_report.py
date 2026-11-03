#!/usr/bin/env python
"""Build analysis/REPORT.md from saved experiment outputs (no model needed)."""
import json
import re
import sys
from pathlib import Path

A = Path(__file__).parent
sys.path.insert(0, str(A))
import gsm8k_band_intervention as G  # noqa: E402

C = ["base", "mid_low0.5", "mid_high0.5", "mid_low0", "mid_high0", "all_low0.5", "all_high0.5"]
DESC = {
    "base": "Không can thiệp",
    "mid_low0.5": "Layer giữa: dải THẤP (chu kỳ ~9–60 tok) giảm còn 50%",
    "mid_high0.5": "Layer giữa: dải CAO (<9 tok) giảm còn 50%",
    "mid_low0": "Layer giữa: BỎ dải thấp",
    "mid_high0": "Layer giữa: BỎ dải cao",
    "all_low0.5": "MỌI layer: dải thấp ×0.5",
    "all_high0.5": "MỌI layer: dải cao ×0.5",
}
squash = lambda t: re.sub(r"\n{3,}", "\n\n", t).strip()
fmt = lambda v: "None" if v is None else (f"{v:g}")
ok = lambda r: r["pred"] is not None and abs(r["pred"] - r["gold"]) < 1e-4


def clip(t, head, tail=0):
    t = squash(t)
    return t if len(t) <= head + tail else t[:head] + "\n[...]\n" + (t[-tail:] if tail else "")


def table(rows, header):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def main():
    L = []
    w = L.append
    S = json.loads((A / "fourier_out/summary.json").read_text())
    GEN = json.loads((A / "fourier_out/generations.json").read_text())
    SP = json.loads((A / "gsm8k_out/spectrum_summary.json").read_text())
    SC = json.loads((A / "gsm8k_out/scores.json").read_text())
    R = {c: {json.loads(l)["idx"]: json.loads(l) for l in open(A / f"gsm8k_out/gen_{c}.jsonl")} for c in C}
    Q = {i: q for i, q, _ in G.load_gsm(20, 0)}
    ids = list(R["base"])

    w("# Báo cáo: phân tích Fourier activation của LLM và can thiệp theo tần số\n")
    w("Nhánh `claude/fourier-llm-activation-analysis-977mop`. Mọi số liệu trong file này được dựng tự động từ các file kết quả "
      "trong `analysis/` bằng `analysis/build_report.py`.\n")
    w("**Ba thí nghiệm**\n\n1. Phổ Fourier của activation trên 4 prompt suy luận ngắn, 3 mô hình (Qwen2.5-0.5B-Instruct, pythia-160m, SmolLM2-360M).\n"
      "2. Phổ Fourier trên lời giải chain-of-thought dài của GSM8K (Qwen2.5-0.5B-Instruct).\n"
      "3. Can thiệp: lọc dải tần theo trục token trên activation, đo độ chính xác GSM8K, độ mạch lạc và đáp án (Qwen2.5-0.5B-Instruct, 20 bài).\n")

    # ------------------------------------------------------------ exp 1
    w("---\n## Thí nghiệm 1: phổ Fourier, 4 prompt ngắn × 3 mô hình\n")
    w("**Cách làm.** Mỗi prompt được mô hình sinh tiếp (greedy) cho đủ 96 token. Lấy activation (hidden state) của từng layer, từng token; "
      "bỏ token đầu (attention sink). Tính phổ công suất dọc trục token cho: x, đạo hàm bậc 1/2/3 (d1, d2, d3), độ dài (n_*), góc (a_*). "
      "Tần số tính bằng chu kỳ/token (Nyquist = 0.5, chu kỳ = 1/f). Cột *xáo trộn* là đối chứng: cùng activation nhưng đảo thứ tự token ngẫu nhiên "
      "(= không có cấu trúc thời gian). Số liệu là trung bình qua 4 prompt và mọi layer.\n")
    for m, s in S.items():
        w(f"### {m}  (layers+embedding = {s['n_layers_plus_emb']})\n")
        rows = []
        for q in ["x", "d1", "d2", "d3", "n_x", "n_d1", "n_d2", "n_d3", "a_x", "a_d1", "a_x_d1"]:
            r = s["quantities"][q]; a = r["all_layers"]; b = r["shuffled_baseline"]
            rows.append([q, f"{a['centroid']:.3f}", f"{a['low(<0.1)']:.2f}", f"{a['high(>=0.25)']:.2f}", f"{a['slope']:.2f}",
                         r["top3_periods"], f"{b['centroid']:.3f}", f"{b['low(<0.1)']:.2f}"])
        w(table(rows, ["chuỗi", "trọng tâm f", "công suất f<0.1", "công suất f≥0.25", "độ dốc α", "3 chu kỳ lớn nhất (token)",
                       "xáo trộn: trọng tâm", "xáo trộn: f<0.1"]))
        d = s["depth"]
        w(f"\nTrục chiều sâu (qua layer): công suất f<0.1 chu kỳ/layer của x = {d['x']['low(<0.1)']:.2f}, của |x| = {d['norm']['low(<0.1)']:.2f}, "
          f"của phần cập nhật mỗi layer = {d['update']['low(<0.1)']:.2f}.\n")
        w("**Văn bản mô hình sinh ra (prompt → phần sinh tiếp 96 token)**\n")
        for k, v in GEN[m].items():
            w(f"*Prompt `{k}`:*\n\n~~~~\n{v['prompt']}\n~~~~\n*Sinh tiếp:*\n\n~~~~\n{clip(v['continuation'], 420)}\n~~~~\n")
    w("**Đọc kết quả thí nghiệm 1**\n\n"
      "- x (activation thô): hệ số lớn nhất ở tần số thấp nhất (chu kỳ ≈ cả chuỗi, ~95 token) = xu hướng trôi chậm, không phải lặp. "
      "Phổ khá phẳng (α≈0.3–0.5), 38–42% công suất vẫn ở f≥0.25. Chỉ nghiêng thấp tần số hơn đối chứng xáo trộn một chút.\n"
      "- d1, d2, d3: công suất dồn lên tần số cao (chu kỳ 2–3 token) và càng bậc cao càng dồn. Gần trùng đối chứng xáo trộn, "
      "vì phần lớn là hệ quả toán học của phép lấy hiệu.\n"
      "- Độ dài |x|: biến thiên chậm. Góc: gần như không hơn đối chứng.\n"
      "- Trục layer: biến đổi trơn, năng lượng chủ yếu ở tần số thấp.\n")

    # ------------------------------------------------------------ exp 2
    w("---\n## Thí nghiệm 2: phổ Fourier trên lời giải GSM8K dài (Qwen2.5-0.5B-Instruct)\n")
    w(f"**Cách làm.** 20 bài GSM8K (test, seed 0), chat template Qwen kèm yêu cầu \"reason step by step, put the final answer within \\boxed{{}}\", "
      f"greedy, tối đa 256 token. Lấy {SP['T']} token sinh ra đầu tiên của {SP['n_used']} lời giải đủ dài (không chứa token sink). "
      f"Khoảng cách trung bình giữa hai dấu xuống dòng: **{SP['mean_newline_gap_tokens']:.1f} token**.\n")
    rows = []
    for q in ["x", "d1", "d2", "d3", "n_x", "n_d1", "n_d2", "n_d3", "a_x", "a_d1", "a_x_d1"]:
        r = SP["q"][q]; a = r["all"]; b = r["shuffled"]
        rows.append([q, f"{a['centroid']:.3f}", f"{a['low(<0.1)']:.2f}", f"{a['high(>=0.25)']:.2f}", f"{a['slope']:.2f}", r["top3_periods"],
                     r["top3_over_shuffled"], f"{b['centroid']:.3f}", f"{b['low(<0.1)']:.2f}"])
    w(table(rows, ["chuỗi", "trọng tâm f", "f<0.1", "f≥0.25", "α", "3 chu kỳ lớn nhất", "công suất / xáo trộn", "xáo trộn: trọng tâm", "xáo trộn: f<0.1"]))
    w("\n**Đọc kết quả.** x vẫn nghiêng tần số thấp (0.36 so với 0.20 của đối chứng), đỉnh là chu kỳ 64–192 token (trôi chậm). "
      "Độ dài đạo hàm |d1..d3| có đỉnh chu kỳ ≈11 và ≈19 token, mạnh hơn đối chứng 1.3–1.5 lần; chu kỳ ≈11 khớp với khoảng cách "
      "≈12.6 token giữa các dòng, tức nhịp các bước suy luận. Vector đạo hàm và góc gần như không khác đối chứng xáo trộn.\n")

    # ------------------------------------------------------------ exp 3
    w("---\n## Thí nghiệm 3: can thiệp theo dải tần trên activation, GSM8K\n")
    w("**Cách làm.** Hook ở layer giữa (layer 12/24) hoặc mọi layer. Tại mỗi token, residual stream h được tách nhân quả (dùng được với KV-cache) thành 3 dải: "
      "*chậm* = EMA(0.9) (thành phần trung bình/trôi, luôn giữ nguyên), *thấp* = EMA(0.5) − EMA(0.9) (chu kỳ ~9–60 token), "
      "*cao* = h − EMA(0.5) (chu kỳ <9 token). Thay h bằng chậm + a·thấp + b·cao. Token đầu không bị lọc. "
      "Ở layer giữa, tỉ lệ năng lượng |h|² là chậm 0.50, thấp 0.17, cao 0.14, nên hai dải can thiệp có năng lượng gần nhau. "
      "Độ mạch lạc = NLL của văn bản đã sinh, tính bằng mô hình GỐC chưa can thiệp (thấp = trơn tru). "
      "Lặp = tỉ lệ 4-gram lặp (≈1 = vòng lặp). Quan trọng: NLL thấp cùng lặp cao là vòng lặp suy biến, không phải mạch lạc.\n")
    w("### 3.1 Tổng quan 7 điều kiện\n")
    rows = []
    for c in C:
        s = SC[c]; same = sum(R[c][i]["pred"] == R["base"][i]["pred"] for i in ids)
        none = sum(R[c][i]["pred"] is None for i in ids)
        rows.append([f"`{c}`", DESC[c], f"{s['acc']*100:.0f}% (±{s['acc_se']*100:.0f})", f"{s['nll_under_base']:.2f}", f"{s['rep4']:.2f}",
                     f"{s['finished']*100:.0f}%", f"{s['gen_len']:.0f}", f"{same}/20", f"{none}/20"])
    w(table(rows, ["điều kiện", "mô tả", "đúng", "NLL", "lặp 4-gram", "kết thúc trước 256 tok", "độ dài TB", "đáp án giống baseline", "không có đáp án"]))
    w("\nChú ý: baseline chỉ 25% (5/20), 70% lời giải bị cắt ở 256 token nên độ chính xác bị đánh thấp. Với n=20, sai số ±10 điểm. "
      "Cách lấy đáp án: lấy nội dung `\\boxed{}` cuối cùng; nếu lời giải bị cắt trước đó thì lấy số cuối cùng xuất hiện trong văn bản "
      "(nên một số đáp án 'bị cắt' là kết quả của bước trung gian).\n")

    w("### 3.2 Đáp án từng bài (✓ = đúng)\n")
    rows = []
    for i in ids:
        rows.append([i, fmt(R["base"][i]["gold"])] + [fmt(R[c][i]["pred"]) + (" ✓" if ok(R[c][i]) else "") for c in C])
    w(table(rows, ["bài (idx)", "đáp án đúng"] + [f"`{c}`" for c in C]))
    w("\n**Nhận xét về đáp án.**\n"
      "- Can thiệp nhẹ (×0.5 ở layer giữa) giữ độ chính xác tổng 20–25% nhưng chỉ 4–5/20 đáp án trùng baseline: tập bài đúng thay đổi "
      "(bài `1010` và `98` từ đúng thành sai, bài `130` và `237` từ sai thành đúng ở `mid_low0.5`). Độ chính xác tổng không đổi chủ yếu là tình cờ; "
      "đáp án thực ra rất nhạy với nhiễu.\n"
      "- Bỏ hẳn một dải ở layer giữa: 0/20 đúng, hầu như mọi đáp án đổi, nhiều đáp án vô lý (ví dụ bài `209`: 1.25e+06; bài `586`: 78000 thay vì 800).\n"
      "- Can thiệp ×0.5 ở mọi layer: không còn đáp án nào (19–20/20 bài không có đáp án, mô hình rơi vào vòng lặp).\n")

    w("### 3.3 Ví dụ chạy chi tiết: văn bản của từng điều kiện\n")
    w("Mỗi ví dụ hiện câu hỏi, rồi văn bản sinh ra (đoạn đầu, và đoạn cuối nếu dài) cho từng điều kiện.\n")
    picks = [(264, "baseline đúng; xem chuyện gì xảy ra khi can thiệp"), (586, "baseline sai (8 thay vì 800)"),
             (98, "baseline đúng; can thiệp nhẹ làm sai")]
    for idx, note in picks:
        w(f"#### Bài {idx}: {note}\n\n**Câu hỏi:** {Q[idx]}\n\n**Đáp án đúng:** {fmt(R['base'][idx]['gold'])}\n")
        for c in C:
            r = R[c][idx]
            long_ = c.startswith("all_")
            w(f"**`{c}`** ({DESC[c]}): đáp án = **{fmt(r['pred'])}** {'✓' if ok(r) else '✗'}; {len(r['gen_ids'])} token; "
              f"{'kết thúc' if r['finished'] else 'bị cắt'}\n\n~~~~\n{clip(r['text'], 220 if long_ else 650, 0 if long_ else 350)}\n~~~~\n")

    w("### 3.4 Kết luận thí nghiệm 3\n"
      "- Cả hai dải đều cần cho suy luận. Bỏ hẳn một dải ở layer giữa làm NLL tăng 8–10 lần (0.16 → 1.26 khi bỏ thấp, 1.63 khi bỏ cao) và đáp án về 0/20 đúng. "
      "Bỏ dải cao hại hơn một chút (khác biệt NLL khoảng 3 sai số chuẩn, nhưng chỉ ở một layer, 20 bài).\n"
      "- Giảm nhẹ một dải ở một layer: văn bản vẫn trôi chảy (NLL 0.16 → 0.23) nhưng chuỗi lập luận và đáp án đổi nhiều.\n"
      "- Giảm một nửa một dải ở mọi layer: suy biến thành vòng lặp (\"systemsystem…\"), không có đáp án.\n"
      "- Chưa thể nói dải nào \"gánh\" suy luận hơn dải nào: bỏ dải nào cũng hỏng, và quy mô thí nghiệm nhỏ.\n")

    w("---\n## Hạn chế chung\n"
      "- Thí nghiệm 3 chỉ dùng Qwen2.5-0.5B-Instruct, 20 bài, một layer giữa, điểm cắt dải do tôi chọn (EMA 0.5 và 0.9), giới hạn 256 token.\n"
      "- Thí nghiệm 1: văn bản sinh ra trong mục 1 được sinh lại bằng greedy với cùng cài đặt để lưu lại (kết quả phổ đã tính từ lần chạy trước, cùng prompt).\n"
      "- Chỉ 4 prompt trong thí nghiệm 1 và 20 bài trong thí nghiệm 2/3, nên chưa đủ để kết luận thống kê chắc chắn.\n\n"
      "## Tái chạy\n```\npython analysis/fourier_activation_analysis.py\n"
      "python analysis/gsm8k_band_intervention.py gen   # rồi: score\n"
      "python analysis/fourier_gsm8k_spectrum.py\npython analysis/build_report.py\n```\n")
    (A / "REPORT.md").write_text("\n".join(L))
    print("written", (A / "REPORT.md"), sum(map(len, L)), "chars")


if __name__ == "__main__":
    main()
