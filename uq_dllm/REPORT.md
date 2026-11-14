# Ước lượng uncertainty của diffusion LLM từ quỹ đạo activation (layer × step)

Nhánh `claude/dllm-uq-activation-trajectory`. Mã trong `uq_dllm/`, kết quả trong `uq_dllm/results/` (vòng 1 và kiểm chứng) và `uq_dllm/results_pooled/` (gộp hai vòng).

## 0. Tóm tắt

**Không phải LLaDA-8B.** Máy này có 15GB RAM, 4 CPU, không GPU; LLaDA-8B (bf16 ≈ 16GB) không nạp được. Tôi dùng `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`: mô hình 0.6B huấn luyện bằng đúng mục tiêu masked-diffusion của LLaDA, sinh theo cùng kiểu LLaDA (semi-autoregressive theo block, low-confidence remasking, greedy). Kết quả dưới đây là của mô hình nhỏ này, chưa phải của LLaDA-8B.

**Câu trả lời ngắn cho "đặc trưng nào liên quan nhất khi model đúng/sai và còn đúng khi mở rộng dữ liệu":**

1. **Logit lens ở layer sâu (layer 20–28) là tín hiệu duy nhất đúng chiều trên cả 3 bộ dữ liệu, ở cả 2 vòng.** Đo bằng độ chắc chắn của phân phối token (max prob, margin, entropy) tính từ activation. AUROC trên mẫu kiểm chứng mới: ARC 0.82–0.83, dịch máy 0.74–0.76, GSM8K 0.63–0.65. Mức này ngang hoặc nhỉnh hơn các baseline độ tin cậy cổ điển (conf_mean 0.57–0.75, ent_max 0.62–0.80), nhưng khoảng tin cậy chồng lấn, nên chưa thể nói hơn hẳn. Đây là tín hiệu "độ tin cậy", không phải thông tin về quỹ đạo.
2. **Đặc trưng thuần quỹ đạo không đủ tin cậy để scale.** Tốt nhất là khoảng cách tương đối đến câu trả lời cuối (`d_ref`) ở layer 15–21 trên GSM8K: AUROC 0.94 ở vòng 1, 0.80 [0.62, 0.95] trên 40 mẫu mới, 0.86 khi gộp 80 mẫu, hơn các baseline (0.58–0.67) trên cùng tập kiểm chứng. Nhưng nó **không chuyển** sang dịch máy hay ARC. Nhóm đặc trưng quỹ đạo "nhất quán giữa các bộ dữ liệu" chọn ở vòng 1 thất bại trên mẫu mới (AUROC 0.65 / 0.58 / 0.49).
3. **Quỹ đạo không thêm thông tin ngoài độ tin cậy.** Hồi quy logistic fit ở vòng 1, đánh giá ở vòng 2: thêm 3 đặc trưng quỹ đạo vào baseline độ tin cậy cho chênh lệch AUROC +0.05 [−0.13, +0.23] (GSM8K), −0.03 [−0.09, +0.03] (dịch máy), −0.10 [−0.20, −0.01] (ARC).
4. **Sàng lọc ~800 đặc trưng trên vài chục mẫu gây ảo tưởng lớn.** AUROC trung bình của top-10 đặc trưng chọn ở vòng 1 tụt từ 0.88 xuống 0.68 (GSM8K) và từ 0.80 xuống 0.61 (dịch máy); riêng ARC giữ nguyên (0.75 → 0.77) vì top-10 toàn là logit lens.

## 1. Thiết lập

| | |
|---|---|
| Mô hình | Qwen3-0.6B-diffusion-mdlm (28 layer, hidden 1024, attention hai chiều). Tự cài lại backbone (`model.py`), khớp tuyệt đối với Qwen3 gốc ở chế độ causal (sai khác 0.0); **không chạy mã từ xa** của repo. |
| Sinh | Semi-AR theo block 32 (ARC: 16), low-confidence remasking, greedy; GSM8K 128 token / 64 bước, dịch máy 64 / 32, ARC 16 / 16. |
| GSM8K | exact match số cuối (`\boxed{}`); 80 mẫu, 24 đúng (30%). |
| Dịch máy | WMT14 de→en (câu 8–30 từ), chrF câu trên đoạn đầu của đầu ra; nhãn "tốt" = chrF ≥ trung vị (38.7); 120 mẫu. |
| ARC-Easy | trắc nghiệm A–D, đầu ra chỉ chữ cái; 200 mẫu, 154 đúng (77%). |
| Vòng 1 / vòng 2 | 40+60+100 mẫu để sàng lọc; 40+60+100 mẫu **hoàn toàn mới** để kiểm chứng (chọn đặc trưng chỉ từ vòng 1). Gộp: 80 / 120 / 200. |

**Quỹ đạo ghi lại:** hidden state `h[s, l, t]` của mọi layer l (kể cả embedding) tại mọi vị trí câu trả lời t, ở mọi bước khử mask s = 0…S−1, cộng một lượt forward cuối trên câu trả lời đã giải mã hết (s = S) làm "đáp án cuối" để so sánh. Chỉ dùng các vị trí trước token kết thúc đầu tiên.

**Thông tin đã thử** (`features.py`; tất cả khoảng cách là tương đối, chia cho norm):
- *Theo bước, từng layer:* độ lớn `|h|`, `|h|/|ref|`; thay đổi so với đáp án cuối `d_ref = |h−ref|/|ref|`, `cos_ref`; vận tốc so với bước liền kề `vel`, `vel_ref`, `cos_prev`; gia tốc `acc`; jerk; góc đổi hướng `turn`; hướng đi có về phía đáp án cuối không `prog`.
- *Cả quỹ đạo:* độ dài đường đi `path`, độ dịch chuyển ròng `net`, `path/net`.
- *Theo chiều sâu (qua layer, tại mỗi bước):* vận tốc, gia tốc, góc đổi hướng giữa layer, khoảng cách và cos tới layer cuối.
- *Logit lens* (layer 8,12,16,20,24,28 × 9 bước × ≤48 vị trí): entropy, max prob, margin, xác suất token cuối, top-1 có trùng token cuối không, KL tới layer cuối, layer nông nhất mà argmax đã trùng token cuối.
- *Baseline cổ điển:* độ tin cậy lúc commit (mean/min/p10), entropy lúc commit, log-lik trung bình, số lần argmax đổi, log-lik token dưới lượt forward cuối, độ dài.
- Mỗi chuỗi được tổng hợp (mean và max trên vị trí) theo nhóm layer (cả, 4 phần tư) và nhóm bước (cả, đầu, giữa, cuối): **802 đặc trưng**.

**Chỉ số và kiểm soát:** AUROC (P[đúng có giá trị cao hơn]), Pearson với nhãn, Spearman với điểm liên tục (chrF), Spearman riêng phần sau khi loại ảnh hưởng của độ dài đầu ra và độ dài prompt; kiểm định hoán vị cho **giá trị lớn nhất của |AUC−0.5| trên mọi đặc trưng** (để hiệu chỉnh việc chọn trong 802 đặc trưng); chia đôi dữ liệu (chọn trên nửa này, đo trên nửa kia); bootstrap 95% CI.

## 2. Vòng 1: sàng lọc (N = 40 / 60 / 100)

Tất cả vượt ngưỡng hoán vị 95% của đặc trưng tốt nhất (GSM8K 0.375, dịch máy 0.272, ARC 0.233; quan sát được 0.437, 0.313, 0.281). Đặc trưng đứng đầu mỗi bộ:

| Bộ dữ liệu | Đặc trưng đứng đầu (AUROC) | Ý nghĩa |
|---|---|---|
| GSM8K | `d_ref` layer 15–21, trung bình mọi bước: 0.06 (tức 0.94 nếu đảo chiều) | Biểu diễn ở layer giữa-sâu càng sớm gần với biểu diễn cuối thì càng hay đúng |
| Dịch máy | `vel`, `vel_ref` các bước đầu, layer 15–28: 0.19 (0.81 đảo chiều) | Các bước đầu thay đổi càng ít thì dịch càng tốt |
| ARC | `lens.maxp`, `lens.margin` layer sâu, các bước đầu: 0.78 | Logit lens ở layer sâu càng chắc thì càng đúng |

Biểu đồ: `results/traj_gsm8k.png`, `traj_wmt.png`, `traj_arc.png` (hàng trên: AUROC theo (layer × step); hàng dưới: quỹ đạo trung bình của nhóm đúng và sai).

## 3. Vòng 2: kiểm chứng trên mẫu mới (đặc trưng đã đóng băng)

AUROC vòng 1 (đã được chọn trên đó, nên lạc quan) → AUROC vòng 2 (trung thực), [95% CI]:

| Bộ dữ liệu | Đặc trưng | Vòng 1 | Vòng 2 |
|---|---|---|---|
| GSM8K | `d_ref` layer 15–21, mọi bước (−) | 0.937 | **0.801** [0.62, 0.95] |
| GSM8K | `cos_ref` layer 15–21, mọi bước (+) | 0.881 | **0.812** [0.66, 0.94] |
| GSM8K | `d_ref` các bước đầu, layer 15–21 (−) | 0.934 | 0.746 |
| GSM8K | `dl_ref` (theo chiều sâu), bước cuối (−) | 0.878 | 0.558 (mất) |
| GSM8K | `path` max (−) | 0.843 | 0.504 (mất) |
| GSM8K | baseline conf_mean / conf_min / ent_mean | 0.71 / 0.71 / 0.75 | 0.66 / 0.58 / 0.67 |
| Dịch máy | `lens.maxp` layer sâu (+) | 0.679 | **0.763** [0.64, 0.88] |
| Dịch máy | `lens.margin` layer sâu (+) | 0.684 | **0.743** [0.61, 0.86] |
| Dịch máy | `vel_ref` các bước đầu, layer 22–28 (−) | 0.806 | 0.639 [0.49, 0.78] |
| Dịch máy | `net` layer 15–21 (−) | 0.780 | 0.677 [0.53, 0.82] |
| Dịch máy | baseline ent_max / flips / conf_min / conf_mean | 0.71 / 0.66 / 0.75 / 0.68 | 0.67 / 0.68 / 0.63 / 0.57 |
| ARC | `lens.margin` layer sâu (+) | 0.760 | **0.832** [0.72, 0.92] |
| ARC | `lens.maxp` layer sâu (+) | 0.751 | **0.823** [0.71, 0.92] |
| ARC | baseline ent_max / conf_min / conf_mean | 0.67 / 0.67 / 0.68 | 0.80 / 0.76 / 0.75 |

AUROC của điểm gộp (trung bình z-score của top-10, chiều cố định theo vòng 1) trên mẫu mới:

| | GSM8K | Dịch máy | ARC |
|---|---|---|---|
| top-10 theo từng bộ | 0.73 | 0.62 | 0.80 |
| top-10 nhất quán giữa các bộ (có cả logit lens) | 0.66 | 0.69 | 0.83 |
| top-10 nhất quán, **chỉ quỹ đạo** | 0.65 | 0.58 | **0.49** |

Giá trị tăng thêm khi thêm quỹ đạo vào baseline (logistic, fit vòng 1, AUROC vòng 2): xem mục 0, điểm 3. Toàn bộ bảng: `results/validation.md`.

## 4. Gộp hai vòng (N = 80 / 120 / 200): sàng lọc lại

- Ngưỡng hoán vị 95% / đặc trưng tốt nhất quan sát được: GSM8K 0.263 / 0.359, dịch máy 0.196 / 0.268, ARC 0.175 / 0.294. Chia đôi (top-5 chọn một nửa, đo nửa kia): AUROC 0.72 / 0.65 / 0.74.
- GSM8K: `d_ref` layer 15–21 là đặc trưng số 1 (AUROC 0.86; Spearman riêng phần với điểm sau khi loại độ dài và độ dài prompt: −0.48). Dịch máy: `cos_prev` max, `cos_ref` max các bước đầu (0.74–0.77), `lens.maxp` (0.73). ARC: toàn logit lens (0.75–0.79).
- Đặc trưng đúng chiều trên cả ba bộ (AUROC mỗi bộ): `lens.ent` layer sâu (0.29 / 0.29 / 0.29, tức 0.71 đảo chiều trên cả ba), `lens.margin` (0.69 / 0.71 / 0.72), `lens.maxp` (0.71 / 0.69 / 0.70). Các baseline `ent_max` (0.33 / 0.30 / 0.28) và `logp_mean` (0.68 / 0.66 / 0.71) ở mức tương đương.
- Kích thước hiệu ứng (GSM8K, nhóm đúng so với sai): `d_ref` 0.459 so với 0.476 (Cohen d = −1.4); `cos_ref` 0.799 so với 0.787 (d = +1.4); vận tốc các bước cuối, layer 14–28: 0.048 so với 0.052 (d = −0.8); góc đổi hướng: 1.46 so với 1.49 (d = −0.65). Chênh lệch tuyệt đối nhỏ (vài phần trăm) nhưng độ phân tán giữa các mẫu cũng rất nhỏ.

## 5. Logit lens theo từng layer (gộp hai vòng)

AUROC của `lens.maxp` / `lens.ent` (trung bình trên 9 bước) theo layer của lens (L28 = phân phối đầu ra thật):

| Bộ dữ liệu | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|
| GSM8K, maxp | 0.59 | 0.61 | 0.58 | 0.66 | 0.60 | 0.67 |
| Dịch máy, maxp | 0.49 | 0.47 | 0.46 | **0.71** | 0.66 | 0.63 |
| ARC, maxp | 0.45 | 0.51 | 0.55 | 0.62 | **0.72** | 0.72 |
| Dịch máy, ent | 0.52 | 0.58 | 0.54 | **0.31** | 0.36 | 0.36 |
| ARC, ent | 0.51 | 0.48 | 0.50 | 0.38 | 0.30 | 0.28 |

- Logit lens ở layer ≤ 16 gần như vô dụng cho dịch máy và ARC (≈ 0.5). Thông tin về đúng/sai chỉ xuất hiện từ khoảng layer 20.
- Ở dịch máy, layer 20 cho AUROC (0.71) cao hơn layer cuối (0.63); ở ARC layer 24 ≈ layer cuối. Tức là phân phối đọc ra từ layer trung gian sâu có thông tin ít nhất bằng đầu ra thật. Mức chênh ở dịch máy chỉ khoảng 1.5 sai số chuẩn, nên đây là gợi ý, chưa phải kết luận.
- Đặc trưng quỹ đạo theo layer (`results_pooled/layerwise.md`): trên GSM8K, `d_ref` có AUROC 0.14–0.18 ở layer 16–24, yếu ở embedding và layer 28; trên dịch máy, vận tốc các bước đầu có AUROC 0.28–0.32 ở layer 14–28; trên ARC không thấy mẫu hình quỹ đạo rõ ràng (AUROC nằm trong khoảng 0.37–0.61). Biểu đồ: `results_pooled/layerwise.png`.

## 6. Kết luận thận trọng

**Có quy luật, giữ được khi mở rộng:**
- Logit-lens confidence ở layer ≥ 20: đúng chiều trên 3/3 bộ ở cả hai vòng, AUROC 0.63–0.83 trên mẫu mới. Đây là ứng viên chính để scale, và ngang các baseline độ tin cậy.

**Có tín hiệu nhưng phụ thuộc bộ dữ liệu:**
- GSM8K: càng sớm gần với biểu diễn cuối ở layer 15–21 (`d_ref` nhỏ, `cos_ref` lớn, độ dịch chuyển ròng nhỏ, ít đổi hướng) thì càng hay đúng; giữ AUROC 0.80 trên mẫu mới và hơn baseline, nhưng chưa chuyển sang dịch máy/ARC.
- Dịch máy: các bước đầu thay đổi ít (vận tốc, gia tốc thấp ở layer 14–28) liên quan đến bản dịch tốt, nhưng giảm mạnh trên mẫu mới (0.81 → 0.60–0.64).

**Không nên tin để scale:** các đặc trưng quỹ đạo chọn theo "nhất quán giữa các bộ dữ liệu" (AUROC mẫu mới 0.65 / 0.58 / 0.49); các đặc trưng theo chiều sâu (`dl_*`) và độ dài đường đi (`path`, `tort`); việc cộng quỹ đạo vào baseline độ tin cậy không cải thiện.

## 7. Hạn chế và điều cần lưu ý

- **Mô hình nhỏ 0.6B, không phải LLaDA-8B.** Cần chạy lại trên GPU với LLaDA thật. `model.py` chỉ cài kiến trúc Qwen3; LLaDA dùng backbone riêng nên phải thêm bộ nạp riêng (hoặc chạy mã chính thức của repo trên máy bạn tin cậy).
- **Cỡ mẫu nhỏ** (N=40–200), nhiều đặc trưng liên quan chặt chẽ với nhau (không độc lập), khoảng tin cậy còn rộng (±0.15–0.2 ở GSM8K và dịch máy).
- Ranh giới block semi-autoregressive làm quỹ đạo có "bước nhảy" cố định (ví dụ bước 16 của dịch máy, bước 48 của GSM8K); nhóm bước đầu/cuối vì thế một phần phản ánh lịch trình block chứ không riêng hành vi mô hình.
- Logit lens chỉ tính ở 6 layer × 9 bước × ≤48 vị trí để giới hạn chi phí; ARC chỉ có 1–5 token đáp án nên quỹ đạo gần như suy biến.
- Nhãn dịch máy là chia theo trung vị chrF (có thể thay bằng COMET); GSM8K chỉ đạt 30% với 0.6B và 128 token, một phần câu trả lời bị cắt.
- Độ dài đầu ra là yếu tố gây nhiễu (AUROC của độ dài: GSM8K 0.35, dịch máy 0.58); các Spearman riêng phần trong bảng đã loại phần này.
- Vòng 2 chỉ dùng một lần để kiểm chứng; sau đó tôi gộp để sàng lọc lại (mục 4), nên mục 4 là khám phá, không còn là kiểm chứng.

## 8. Chạy lại

```
python uq_dllm/run.py --dataset gsm8k --n 40 --out uq_dllm/out      # wmt, arc tương tự; --offset để lấy mẫu mới
python uq_dllm/analyze.py --out uq_dllm/results                      # sàng lọc + hoán vị + chia đôi + biểu đồ
python uq_dllm/validate.py --train uq_dllm/out --test uq_dllm/out2   # kiểm chứng đặc trưng đã đóng băng
python uq_dllm/layerwise.py --root uq_dllm/pooled                    # AUROC theo layer, logit lens
```
Dữ liệu quỹ đạo thô (`uq_dllm/out*/`, `uq_dllm/pooled/`) được bỏ khỏi git vì dung lượng lớn.

**Bước tiếp theo đề xuất:** (1) chạy LLaDA-8B trên GPU với ≥ 500 mẫu mỗi bộ; (2) đăng ký trước một danh sách nhỏ (logit lens layer sâu; `d_ref` layer giữa-sâu cho suy luận) thay vì sàng lọc 800 đặc trưng; (3) thêm bộ dữ liệu thứ tư để kiểm tra `d_ref` có chuyển được giữa các tác vụ suy luận hay không; (4) tăng số bước khử mask cho phần giải mã để quỹ đạo dài hơn.
