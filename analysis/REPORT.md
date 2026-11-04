# Báo cáo: phân tích Fourier activation của LLM và can thiệp theo tần số

Nhánh `claude/fourier-llm-activation-analysis-977mop`. Mọi số liệu trong file này được dựng tự động từ các file kết quả trong `analysis/` bằng `analysis/build_report.py`.

**Ba thí nghiệm**

1. Phổ Fourier của activation trên 4 prompt suy luận ngắn, 3 mô hình (Qwen2.5-0.5B-Instruct, pythia-160m, SmolLM2-360M).
2. Phổ Fourier trên lời giải chain-of-thought dài của GSM8K (Qwen2.5-0.5B-Instruct).
3. Can thiệp: lọc dải tần theo trục token trên activation, đo độ chính xác GSM8K, độ mạch lạc và đáp án (Qwen2.5-0.5B-Instruct, 20 bài).
4. Nén KV-cache của phần reasoning bằng DCT (Fourier) và hai đối chứng, đo độ chính xác, độ mạch lạc và đáp án (Qwen2.5-0.5B-Instruct, 20 bài).

---
## Thí nghiệm 1: phổ Fourier, 4 prompt ngắn × 3 mô hình

**Cách làm.** Mỗi prompt được mô hình sinh tiếp (greedy) cho đủ 96 token. Lấy activation (hidden state) của từng layer, từng token; bỏ token đầu (attention sink). Tính phổ công suất dọc trục token cho: x, đạo hàm bậc 1/2/3 (d1, d2, d3), độ dài (n_*), góc (a_*). Tần số tính bằng chu kỳ/token (Nyquist = 0.5, chu kỳ = 1/f). Cột *xáo trộn* là đối chứng: cùng activation nhưng đảo thứ tự token ngẫu nhiên (= không có cấu trúc thời gian). Số liệu là trung bình qua 4 prompt và mọi layer.

### Qwen/Qwen2.5-0.5B-Instruct  (layers+embedding = 25)

| chuỗi | trọng tâm f | công suất f<0.1 | công suất f≥0.25 | độ dốc α | 3 chu kỳ lớn nhất (token) | xáo trộn: trọng tâm | xáo trộn: f<0.1 |
|---|---|---|---|---|---|---|---|
| x | 0.203 | 0.33 | 0.38 | 0.39 | [95.0, 23.75, 31.67] | 0.254 | 0.19 |
| d1 | 0.335 | 0.02 | 0.78 | -1.31 | [2.88, 2.97, 2.02] | 0.352 | 0.01 |
| d2 | 0.376 | 0.00 | 0.91 | -2.67 | [2.02, 2.21, 2.26] | 0.385 | 0.00 |
| d3 | 0.397 | 0.00 | 0.96 | -3.81 | [2.02, 2.21, 2.16] | 0.403 | 0.00 |
| n_x | 0.199 | 0.35 | 0.37 | 0.41 | [95.0, 47.5, 19.0] | 0.254 | 0.18 |
| n_d1 | 0.197 | 0.28 | 0.32 | 0.42 | [19.0, 5.59, 23.75] | 0.205 | 0.28 |
| n_d2 | 0.162 | 0.36 | 0.21 | 0.79 | [19.0, 23.75, 6.33] | 0.164 | 0.36 |
| n_d3 | 0.145 | 0.41 | 0.15 | 1.06 | [23.75, 19.0, 6.33] | 0.146 | 0.40 |
| a_x | 0.194 | 0.31 | 0.33 | 0.44 | [95.0, 47.5, 6.79] | 0.213 | 0.27 |
| a_d1 | 0.285 | 0.15 | 0.64 | -0.24 | [2.97, 2.88, 2.44] | 0.275 | 0.16 |
| a_x_d1 | 0.272 | 0.15 | 0.56 | -0.13 | [2.97, 5.59, 2.88] | 0.285 | 0.13 |

Trục chiều sâu (qua layer): công suất f<0.1 chu kỳ/layer của x = 0.88, của |x| = 0.99, của phần cập nhật mỗi layer = 0.65.

**Văn bản mô hình sinh ra (prompt → phần sinh tiếp 96 token)**

*Prompt `arith`:*

~~~~
Question: Tom has 3 apples and buys 4 more, then gives 2 away. How many apples does he have? Let's think step by step.
Answer:
~~~~
*Sinh tiếp:*

~~~~
Step 1: Determine the initial number of apples Tom had.
Tom starts with 3 apples.

Step 2: Calculate the total number of apples Tom bought.
Tom buys 4 more apples, so we add these to his initial amount:
3 + 4 = 7

Step 3:
~~~~

*Prompt `logic`:*

~~~~
All birds can fly. A penguin is a bird. Does a penguin fly according to these premises? Reason step by step.
Answer:
~~~~
*Sinh tiếp:*

~~~~
To determine whether a penguin flies based on the given premises, we need to carefully analyze each premise and see if it logically leads to the conclusion that "a penguin flies."

1. **Premise 1:** Birds can fly.
   - This premise states that all birds are capable of flying.

2. **Premise
~~~~

*Prompt `seq`:*

~~~~
Continue the pattern and explain: 2, 4, 6, 8, ... The next number is
~~~~
*Sinh tiếp:*

~~~~
____
A. 10  
B. 12  
C. 14  
D. 16

To determine the next number in the sequence 2, 4, 6, 8, ..., we need to identify the pattern governing the sequence. Let's examine the differences between consecutive terms:

- The difference between the second term
~~~~

*Prompt `story`:*

~~~~
Anna put her keys in the drawer, then went to the garden. Bob moved the keys to the table. Where will Anna look for her keys first? Think carefully.
Answer:
~~~~
*Sinh tiếp:*

~~~~
To determine where Anna will look for her keys after Bob moves them to the table, we need to consider the sequence of events and the logical order in which people typically interact with objects.

1. **Initial Position**: Anna initially placed her keys in a drawer.
2. **Bob's Action**: Bob
~~~~

### EleutherAI/pythia-160m  (layers+embedding = 13)

| chuỗi | trọng tâm f | công suất f<0.1 | công suất f≥0.25 | độ dốc α | 3 chu kỳ lớn nhất (token) | xáo trộn: trọng tâm | xáo trộn: f<0.1 |
|---|---|---|---|---|---|---|---|
| x | 0.220 | 0.30 | 0.42 | 0.27 | [95.0, 5.59, 10.56] | 0.252 | 0.20 |
| d1 | 0.347 | 0.02 | 0.80 | -1.46 | [2.21, 2.02, 2.16] | 0.352 | 0.01 |
| d2 | 0.386 | 0.00 | 0.92 | -2.82 | [2.21, 2.02, 2.16] | 0.385 | 0.00 |
| d3 | 0.406 | 0.00 | 0.97 | -3.92 | [2.21, 2.02, 2.16] | 0.403 | 0.00 |
| n_x | 0.215 | 0.32 | 0.42 | 0.22 | [95.0, 5.28, 10.56] | 0.239 | 0.24 |
| n_d1 | 0.147 | 0.45 | 0.22 | 0.83 | [95.0, 10.56, 47.5] | 0.161 | 0.38 |
| n_d2 | 0.114 | 0.54 | 0.12 | 1.34 | [95.0, 10.56, 47.5] | 0.127 | 0.48 |
| n_d3 | 0.098 | 0.61 | 0.08 | 1.63 | [95.0, 10.56, 47.5] | 0.113 | 0.53 |
| a_x | 0.192 | 0.33 | 0.34 | 0.48 | [95.0, 10.56, 47.5] | 0.221 | 0.25 |
| a_d1 | 0.236 | 0.29 | 0.50 | 0.13 | [95.0, 47.5, 2.21] | 0.281 | 0.16 |
| a_x_d1 | 0.291 | 0.12 | 0.65 | -0.29 | [2.21, 3.65, 2.71] | 0.298 | 0.11 |

Trục chiều sâu (qua layer): công suất f<0.1 chu kỳ/layer của x = 0.74, của |x| = 0.84, của phần cập nhật mỗi layer = 0.64.

**Văn bản mô hình sinh ra (prompt → phần sinh tiếp 96 token)**

*Prompt `arith`:*

~~~~
Question: Tom has 3 apples and buys 4 more, then gives 2 away. How many apples does he have? Let's think step by step.
Answer:
~~~~
*Sinh tiếp:*

~~~~
Tom has 3 apples and buys 4 more, then gives 2 away. How many apples does he have? Let's think step by step.
Answer:

Tom has 3 apples and buys 4 more, then gives 2 away. How many apples does he have? Let's think step by step.
~~~~

*Prompt `logic`:*

~~~~
All birds can fly. A penguin is a bird. Does a penguin fly according to these premises? Reason step by step.
Answer:
~~~~
*Sinh tiếp:*

~~~~
Yes.

A:

The answer is yes.

A:

The answer is no.

A:

The answer is yes.

A:

The answer is no.

A:

The answer is yes.

A:

The answer is
~~~~

*Prompt `seq`:*

~~~~
Continue the pattern and explain: 2, 4, 6, 8, ... The next number is
~~~~
*Sinh tiếp:*

~~~~
the number of times the pattern is repeated.

A:

You can use the following code:

$pattern = '2,4,6,8,10,12,14,16,18,20,22,24,28,30,32,34,36,38,40,42,44,46,48,50,52
~~~~

*Prompt `story`:*

~~~~
Anna put her keys in the drawer, then went to the garden. Bob moved the keys to the table. Where will Anna look for her keys first? Think carefully.
Answer:
~~~~
*Sinh tiếp:*

~~~~
"I'm going to get the keys."

"Yes, I know."

"I'll get them."

"Yes, I'll get them."

"Yes, I'll get them."

"Yes, I'll get them."
~~~~

### HuggingFaceTB/SmolLM2-360M  (layers+embedding = 33)

| chuỗi | trọng tâm f | công suất f<0.1 | công suất f≥0.25 | độ dốc α | 3 chu kỳ lớn nhất (token) | xáo trộn: trọng tâm | xáo trộn: f<0.1 |
|---|---|---|---|---|---|---|---|
| x | 0.206 | 0.32 | 0.38 | 0.36 | [95.0, 15.83, 47.5] | 0.254 | 0.19 |
| d1 | 0.333 | 0.02 | 0.77 | -1.33 | [2.64, 2.71, 2.32] | 0.351 | 0.01 |
| d2 | 0.374 | 0.00 | 0.91 | -2.72 | [2.64, 2.32, 2.02] | 0.385 | 0.00 |
| d3 | 0.395 | 0.00 | 0.96 | -3.84 | [2.02, 2.32, 2.64] | 0.403 | 0.00 |
| n_x | 0.211 | 0.29 | 0.39 | 0.32 | [15.83, 47.5, 95.0] | 0.252 | 0.19 |
| n_d1 | 0.197 | 0.28 | 0.34 | 0.45 | [5.28, 95.0, 15.83] | 0.199 | 0.29 |
| n_d2 | 0.156 | 0.37 | 0.20 | 0.94 | [95.0, 47.5, 15.83] | 0.154 | 0.38 |
| n_d3 | 0.145 | 0.40 | 0.16 | 1.12 | [95.0, 47.5, 15.83] | 0.140 | 0.43 |
| a_x | 0.225 | 0.19 | 0.43 | 0.20 | [5.28, 95.0, 3.17] | 0.222 | 0.24 |
| a_d1 | 0.290 | 0.10 | 0.62 | -0.33 | [2.02, 3.96, 2.71] | 0.275 | 0.16 |
| a_x_d1 | 0.305 | 0.06 | 0.69 | -0.55 | [2.71, 2.79, 2.97] | 0.317 | 0.07 |

Trục chiều sâu (qua layer): công suất f<0.1 chu kỳ/layer của x = 0.90, của |x| = 0.97, của phần cập nhật mỗi layer = 0.21.

**Văn bản mô hình sinh ra (prompt → phần sinh tiếp 96 token)**

*Prompt `arith`:*

~~~~
Question: Tom has 3 apples and buys 4 more, then gives 2 away. How many apples does he have? Let's think step by step.
Answer:
~~~~
*Sinh tiếp:*

~~~~
Tom has 3 apples and buys 4 more, then gives 2 away.

Step 1:

Let's think about the problem.

Step 2:

Let's think about the problem.

Step 3:

Let's think about the
~~~~

*Prompt `logic`:*

~~~~
All birds can fly. A penguin is a bird. Does a penguin fly according to these premises? Reason step by step.
Answer:
~~~~
*Sinh tiếp:*

~~~~
No.

Question 10:

A man is standing on a bridge. He sees a boat moving towards him. He sees that the boat is moving towards him at a speed of 10 km/hr. He also sees that the boat is moving towards him at a speed of 10 km/hr.
~~~~

*Prompt `seq`:*

~~~~
Continue the pattern and explain: 2, 4, 6, 8, ... The next number is
~~~~
*Sinh tiếp:*

~~~~
10.

• 10 is the 3rd number in the sequence.

• 10 is the 4th number in the sequence.

• 10 is the 6th number in the sequence.

• 10 is the 8th number in the sequence.

• 1
~~~~

*Prompt `story`:*

~~~~
Anna put her keys in the drawer, then went to the garden. Bob moved the keys to the table. Where will Anna look for her keys first? Think carefully.
Answer:
~~~~
*Sinh tiếp:*

~~~~
Anna will look for her keys in the drawer.

Question 2.
What is the next number in the sequence? 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1
~~~~

**Đọc kết quả thí nghiệm 1**

- x (activation thô): hệ số lớn nhất ở tần số thấp nhất (chu kỳ ≈ cả chuỗi, ~95 token) = xu hướng trôi chậm, không phải lặp. Phổ khá phẳng (α≈0.3–0.5), 38–42% công suất vẫn ở f≥0.25. Chỉ nghiêng thấp tần số hơn đối chứng xáo trộn một chút.
- d1, d2, d3: công suất dồn lên tần số cao (chu kỳ 2–3 token) và càng bậc cao càng dồn. Gần trùng đối chứng xáo trộn, vì phần lớn là hệ quả toán học của phép lấy hiệu.
- Độ dài |x|: biến thiên chậm. Góc: gần như không hơn đối chứng.
- Trục layer: biến đổi trơn, năng lượng chủ yếu ở tần số thấp.

---
## Thí nghiệm 2: phổ Fourier trên lời giải GSM8K dài (Qwen2.5-0.5B-Instruct)

**Cách làm.** 20 bài GSM8K (test, seed 0), chat template Qwen kèm yêu cầu "reason step by step, put the final answer within \boxed{}", greedy, tối đa 256 token. Lấy 192 token sinh ra đầu tiên của 19 lời giải đủ dài (không chứa token sink). Khoảng cách trung bình giữa hai dấu xuống dòng: **12.6 token**.

| chuỗi | trọng tâm f | f<0.1 | f≥0.25 | α | 3 chu kỳ lớn nhất | công suất / xáo trộn | xáo trộn: trọng tâm | xáo trộn: f<0.1 |
|---|---|---|---|---|---|---|---|---|
| x | 0.195 | 0.36 | 0.35 | 0.41 | [64.0, 96.0, 192.0] | [2.47, 2.37, 2.62] | 0.253 | 0.20 |
| d1 | 0.331 | 0.03 | 0.76 | -1.31 | [2.21, 2.06, 2.18] | [0.94, 0.9, 0.9] | 0.353 | 0.01 |
| d2 | 0.376 | 0.00 | 0.90 | -2.78 | [2.21, 2.06, 2.04] | [1.01, 0.96, 0.94] | 0.387 | 0.00 |
| d3 | 0.398 | 0.00 | 0.96 | -4.12 | [2.06, 2.21, 2.04] | [0.99, 1.04, 0.97] | 0.405 | 0.00 |
| n_x | 0.209 | 0.33 | 0.39 | 0.32 | [192.0, 64.0, 96.0] | [2.89, 2.2, 2.18] | 0.252 | 0.20 |
| n_d1 | 0.206 | 0.30 | 0.37 | 0.31 | [11.29, 10.67, 19.2] | [1.5, 1.43, 1.31] | 0.207 | 0.28 |
| n_d2 | 0.159 | 0.40 | 0.22 | 0.74 | [11.29, 10.67, 19.2] | [1.51, 1.49, 1.31] | 0.160 | 0.37 |
| n_d3 | 0.140 | 0.45 | 0.16 | 0.99 | [11.29, 19.2, 10.67] | [1.48, 1.32, 1.5] | 0.144 | 0.42 |
| a_x | 0.223 | 0.26 | 0.42 | 0.19 | [11.29, 19.2, 24.0] | [1.35, 1.25, 1.27] | 0.221 | 0.25 |
| a_d1 | 0.279 | 0.18 | 0.60 | -0.14 | [2.43, 2.59, 2.21] | [1.42, 1.45, 1.26] | 0.271 | 0.18 |
| a_x_d1 | 0.281 | 0.14 | 0.61 | -0.24 | [2.09, 2.11, 3.2] | [1.37, 1.27, 1.38] | 0.282 | 0.14 |

**Đọc kết quả.** x vẫn nghiêng tần số thấp (0.36 so với 0.20 của đối chứng), đỉnh là chu kỳ 64–192 token (trôi chậm). Độ dài đạo hàm |d1..d3| có đỉnh chu kỳ ≈11 và ≈19 token, mạnh hơn đối chứng 1.3–1.5 lần; chu kỳ ≈11 khớp với khoảng cách ≈12.6 token giữa các dòng, tức nhịp các bước suy luận. Vector đạo hàm và góc gần như không khác đối chứng xáo trộn.

---
## Thí nghiệm 3: can thiệp theo dải tần trên activation, GSM8K

**Cách làm.** Hook ở layer giữa (layer 12/24) hoặc mọi layer. Tại mỗi token, residual stream h được tách nhân quả (dùng được với KV-cache) thành 3 dải: *chậm* = EMA(0.9) (thành phần trung bình/trôi, luôn giữ nguyên), *thấp* = EMA(0.5) − EMA(0.9) (chu kỳ ~9–60 token), *cao* = h − EMA(0.5) (chu kỳ <9 token). Thay h bằng chậm + a·thấp + b·cao. Token đầu không bị lọc. Ở layer giữa, tỉ lệ năng lượng |h|² là chậm 0.50, thấp 0.17, cao 0.14, nên hai dải can thiệp có năng lượng gần nhau. Độ mạch lạc = NLL của văn bản đã sinh, tính bằng mô hình GỐC chưa can thiệp (thấp = trơn tru). Lặp = tỉ lệ 4-gram lặp (≈1 = vòng lặp). Quan trọng: NLL thấp cùng lặp cao là vòng lặp suy biến, không phải mạch lạc.

### 3.1 Tổng quan 7 điều kiện

| điều kiện | mô tả | đúng | NLL | lặp 4-gram | kết thúc trước 256 tok | độ dài TB | đáp án giống baseline | không có đáp án |
|---|---|---|---|---|---|---|---|---|
| `base` | Không can thiệp | 25% (±10) | 0.16 | 0.20 | 30% | 245 | 20/20 | 0/20 |
| `mid_low0.5` | Layer giữa: dải THẤP (chu kỳ ~9–60 tok) giảm còn 50% | 25% (±10) | 0.23 | 0.16 | 40% | 229 | 4/20 | 0/20 |
| `mid_high0.5` | Layer giữa: dải CAO (<9 tok) giảm còn 50% | 20% (±9) | 0.23 | 0.18 | 30% | 246 | 5/20 | 0/20 |
| `mid_low0` | Layer giữa: BỎ dải thấp | 0% (±0) | 1.26 | 0.22 | 55% | 227 | 2/20 | 0/20 |
| `mid_high0` | Layer giữa: BỎ dải cao | 0% (±0) | 1.63 | 0.25 | 20% | 245 | 1/20 | 0/20 |
| `all_low0.5` | MỌI layer: dải thấp ×0.5 | 0% (±0) | 0.21 | 0.99 | 0% | 256 | 0/20 | 20/20 |
| `all_high0.5` | MỌI layer: dải cao ×0.5 | 0% (±0) | 1.31 | 0.93 | 5% | 244 | 0/20 | 19/20 |

Chú ý: baseline chỉ 25% (5/20), 70% lời giải bị cắt ở 256 token nên độ chính xác bị đánh thấp. Với n=20, sai số ±10 điểm. Cách lấy đáp án: lấy nội dung `\boxed{}` cuối cùng; nếu lời giải bị cắt trước đó thì lấy số cuối cùng xuất hiện trong văn bản (nên một số đáp án 'bị cắt' là kết quả của bước trung gian).

### 3.2 Đáp án từng bài (✓ = đúng)

| bài (idx) | đáp án đúng | `base` | `mid_low0.5` | `mid_high0.5` | `mid_low0` | `mid_high0` | `all_low0.5` | `all_high0.5` |
|---|---|---|---|---|---|---|---|---|
| 264 | 400 | 400 ✓ | 400 ✓ | 400 ✓ | 22 | 10 | None | None |
| 725 | 25 | 5 | 2 | 20 | 3 | 55 | None | None |
| 996 | 3 | 7 | 20 | 4 | 5 | 2 | None | None |
| 586 | 800 | 8 | 200 | 80 | 200 | 78000 | None | None |
| 1010 | 5 | 5 ✓ | 7 | 5 ✓ | 28 | 2 | None | None |
| 12 | 13 | 90 | 3 | 4 | 6 | 12 | None | None |
| 904 | 8 | 8 ✓ | 8 ✓ | 2 | 7 | 12 | None | None |
| 130 | 30 | 10 | 30 ✓ | 30 ✓ | 10 | 70 | None | None |
| 209 | 145 | 1500 | 120 | 1200 | 12 | 1.25e+06 | None | None |
| 244 | 20 | 2 | 27 | 36 | 2 | 3 | None | None |
| 237 | 90 | 120 | 90 ✓ | 120 | 40 | 6 | None | None |
| 689 | 7 | 3 | 4 | 10 | 0.4 | 0 | None | None |
| 601 | 104 | 48 | 2 | 12 | 13 | 17 | None | None |
| 98 | 5 | 5 ✓ | 25 | 15 | 30 | 42 | None | 1.11111e+226 |
| 1284 | 25 | 2 | 20 | 18 | 1 | 6 | None | None |
| 1131 | 96 | 120 | 1.2 | 1.2 | 1.2 | 3 | None | None |
| 1272 | 296 | 0.7 | 56 | 112 | 64 | 1 | None | None |
| 482 | 26 | 26 ✓ | 26 ✓ | 26 ✓ | 20 | 170 | None | None |
| 992 | 52 | 12 | 16 | 12 | 48 | 12 | None | None |
| 454 | 150 | 240 | 240 | 24 | 12 | 24 | None | None |

**Nhận xét về đáp án.**
- **Cảnh báo về mức nhiễu nền (đo ở thí nghiệm 4):** chạy cùng mô hình, cùng prompt, KHÔNG can thiệp, chỉ khác đường tính số (`generate` của HF so với vòng sinh tự viết) đã cho: chỉ 5/20 đáp án trùng, 0/20 văn bản trùng, tiền tố chung trung bình 28 token, độ chính xác 25% so với 30%. Vì vậy việc ở can thiệp nhẹ (×0.5) chỉ 4–5/20 đáp án trùng baseline NẰM NGANG mức nhiễu nền, không chứng minh được can thiệp nhẹ làm đổi đáp án; greedy decoding tự nó nhạy với sai khác số học rất nhỏ. Chênh lệch độ chính xác 20–25% so với 25% cũng nằm trong sai số.
- Bỏ hẳn một dải ở layer giữa: 0/20 đúng, hầu như mọi đáp án đổi, nhiều đáp án vô lý (ví dụ bài `209`: 1.25e+06; bài `586`: 78000 thay vì 800).
- Can thiệp ×0.5 ở mọi layer: không còn đáp án nào (19–20/20 bài không có đáp án, mô hình rơi vào vòng lặp).

### 3.3 Ví dụ chạy chi tiết: văn bản của từng điều kiện

Mỗi ví dụ hiện câu hỏi, rồi văn bản sinh ra (đoạn đầu, và đoạn cuối nếu dài) cho từng điều kiện.

#### Bài 264: baseline đúng; xem chuyện gì xảy ra khi can thiệp

**Câu hỏi:** Carmen has $100, Samantha has $25 more than Carmen, and Daisy has $50 more than Samantha. How much do all three girls have combined?

**Đáp án đúng:** 400

**`base`** (Không can thiệp): đáp án = **400** ✓; 256 token; bị cắt

~~~~
To determine the total amount of money that Carmen, Samantha, and Daisy have combined, we will follow these steps:

1. **Determine how much money Samantha has:**
   - Carmen has $100.
   - Samantha has $25 more than Carmen.
   - Therefore, Samantha's amount is:
     \[
     100 + 25 = 125 \text{ dollars}
     \]

2. **Determine how much money Daisy has:**
   - Samantha has $125.
   - Daisy has $50 more than Samantha.
   - Therefore, Daisy's amount is:
     \[
     125 + 50 = 175 \text{ dollars}
     \]

3. **Calculate the total amount of money all three girls have combined:**
   - Carmen has $100.
   - Samantha has $125.
   - Daisy has $175.
   - Adding these amounts together gives:
     \[
     100 + 125 + 175 = 400 \text{ dollars}
     \]

Thus, the total amount of money that Carmen, Samantha, and Daisy have combined is \(\
~~~~

**`mid_low0.5`** (Layer giữa: dải THẤP (chu kỳ ~9–60 tok) giảm còn 50%): đáp án = **400** ✓; 213 token; kết thúc

~~~~
To determine the total amount of money that Carmen, Samantha, and Daisy have together, we will follow these steps:

1. Determine how much money each person has.
2. Sum up the amounts for Carmen, Samantha, and Daisy.

First, let's find out how much money Samantha has:
- Carmen has $100.
- Samantha has $25 more than Carmen.
So, Samantha's amount is:
\[ 100 + 25 = 125 \]

Next, we need to find out how much money Daisy has:
- Daisy has $50 more than Samantha.
So, Daisy's amount is:
\[ 125 + 50 = 175 \]

Now, we can sum up the amounts of money that Carmen, Samantha, and Daisy have:
\[ 100 + 125 + 175 = 400 \]

Therefore, the total amount of money that all three girls have combined is:
\[ \boxed{400} \]
~~~~

**`mid_high0.5`** (Layer giữa: dải CAO (<9 tok) giảm còn 50%): đáp án = **400** ✓; 248 token; kết thúc

~~~~
To determine the total amount of money that Carmen, Samantha, and Daisy have together, we will follow these steps:

1. **Determine how much money Samantha has:**
   - We know that Samantha has $25 more than what Carmen has.
   - Carmen has $100.
   - Therefore, Samantha's amount = \( 100 + 25 = 125 \) dollars.

2. **Determine how much money Daisy has:**
   - We know that Daisy has $50 more than what Samantha has.
   - Samantha has $125.
   - Therefore, Daisy's amount = \( 125 + 50 = 175 \) dollars.

3. **Calculate the total amount of money all three girls have together:**
   - Carmen has $100.
   - Samantha has $125.
   - Daisy has $175.
   - Total amount = \( 100 + 125 + 175 = 400 \) dollars.

Thus, the total amount of money that Carmen, Samantha, and Daisy have combined is \(\boxed{400}\).
~~~~

**`mid_low0`** (Layer giữa: BỎ dải thấp): đáp án = **22** ✗; 167 token; kết thúc

~~~~
To determine how much money Carmen has, we start with the amount of money that Carmen has. According to the problem, Carmen has $25.

So, Carmen has:
\[ \$25.

Next, we need to find out how much money Samantha has. The problem states that Samantha has $50 more than what Carmen has. Therefore, Samantha has:
\[ $50.
\[ \$50.

Now, we add the amount that Samantha has to the amount that Carmen has:
\[ \$25 + $50 = $75.

Finally, we add the amounts that both Carmen and Samantha have together:
\[ $75 + $50 = $22.

Therefore, the total is $22.

The sum of all three amounts is $\boxed{22.}

$22.
~~~~

**`mid_high0`** (Layer giữa: BỎ dải cao): đáp án = **10** ✗; 256 token; bị cắt

~~~~
Step 1: First, let's find out how how much money each person has:
 - Carmen has $$100.

 - Since Carmen has $$2 more than than what Carmen has,, we can subtract $$2 to get the amount of money Carmen that Carmen has has.

 So, Carmen has $$$100 + ($$2) = $$$102

 - Now, we need to find out how much money Carmen has has:

 - Carmen has $$2 more than than what what Carmen has has, so, we subtract $$2 to get the amount of money money Carmen that Carmen has has has.

 So, Carmen has $$$$100 + ($($2)) = $$$102

 - Now, we need to find out how how much money Carmen has has:

 - Carmen has $$2 more than than what what Carmen has has, so, we subtract $$2 to get the amount of money money money Carmen that Carmen has has has has.

 So, Carmen has $$$$100 + ($($($2)) = $$$102

 - Now We know how how much money Carmen has has:

 - Carmen has $$$10
~~~~

**`all_low0.5`** (MỌI layer: dải thấp ×0.5): đáp án = **None** ✗; 256 token; bị cắt

~~~~
systemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsyst
[...]

~~~~

**`all_high0.5`** (MỌI layer: dải cao ×0.5): đáp án = **None** ✗; 256 token; bị cắt

~~~~
?

?

?

AssistantAssistantAssistantAssistantAssistantAssistantAssistantAssistantAssistantAssistant

AssistantAssistantAssistantAssistantAssistantAssistantAssistantAssistantAssistantAssistantAssistantAssistantAssistantAs
[...]

~~~~

#### Bài 586: baseline sai (8 thay vì 800)

**Câu hỏi:** Hannah needs to drink 100 ml of water for every 200 calories she burns. She spends 2 hours doing aerobics, which burns 500 calories/hour, and 1 hour running, which burns 600 calories/hour. How many ml of water does she need to drink?

**Đáp án đúng:** 800

**`base`** (Không can thiệp): đáp án = **8** ✗; 256 token; bị cắt

~~~~
To determine how much water Hannah needs to drink, we first need to calculate the total number of calories she burns from her exercise.

First, we calculate the calories burned from running:
\[ 1 \text{ hour} \times 600 \text{ calories/hour} = 600 \text{ calories} \]

Next, we calculate the calories burned from aerobics:
\[ 2 \text{ hours} \times 500 \text{ calories/hour} = 1000 \text{ calories} \]

Now, we add the calories burned from both activities to find the total calories burned:
\[ 600 \text{ calories} + 1000 \text{ calories} = 1600 \text{ calories} \]

According to the problem, Hannah needs to drink 100 ml of water for every 200 calories burned. We can now calculate the amount of water needed based on the total calories burned:
\[ \frac{1600 \text{ calories}}{200 \text{ calories/ml}} = 8 \text{ ml} \]

Therefore, the amount of water Hannah needs to drink is:
\[
\boxed
~~~~

**`mid_low0.5`** (Layer giữa: dải THẤP (chu kỳ ~9–60 tok) giảm còn 50%): đáp án = **200** ✗; 256 token; bị cắt

~~~~
To determine how much water Hannah needs to drink, we can break down the problem into a few steps:

1. Calculate the total amount of calories burned.
2. Determine the total amount of water needed based on the calorie burn.

First, let's calculate the calories burned from running:
- Running burns 600 calories per hour.
- Hannah runs for 1 hour: \(600 \text{ calories} \times 1 \text{ hour} = 600 \text{ calories}\).

Next, let's calculate the calories burned from aerobics:
- Aerobics burns 500 calories per hour.
- Hannah spends 2 hours doing aerobics: \(500 \text{ calories} \times 2 \text{ hours} = 1000 \text{ calories}\).

Now, add the calories burned from both activities to get the total calories burned:
\[600 \text{ calories (from running)} + 1000 \text{ calories (from aerobics)} = 1600 \text{ calories}\].

According to the problem, Hannah needs to drink 100 ml of water for every 200 calories burned. We need to
~~~~

**`mid_high0.5`** (Layer giữa: dải CAO (<9 tok) giảm còn 50%): đáp án = **80** ✗; 256 token; bị cắt

~~~~
To determine how much water Hannah needs to drink, we first need to calculate the total number of calories she burns from her exercise.

First, we calculate the calories burned from running:
\[ 1 \text{ hour} \times 600 \text{ calories/hour} = 600 \text{ calories} \]

Next, we calculate the calories burned from doing aerobics:
\[ 2 \text{ hours} \times 500 \text{ calories/hour} = 1000 \text{ calories} \]

Now, we add the calories burned from both activities to find the total caloric intake:
\[ 600 \text{ calories} + 1000 \text{ calories} = 1600 \text{ calories} \]

According to the problem, Hannah drinks 100 ml of water for every 200 calories burned. We can now calculate the amount of water needed based on the total caloric intake:
\[ \frac{1600 \text{ calories}}{200 \text{ calories/100 ml}} = 80 \text{ ml} \]

Therefore, the number of milliliters of water
~~~~

**`mid_low0`** (Layer giữa: BỎ dải thấp): đáp án = **200** ✗; 256 token; bị cắt

~~~~
To determine how much water Hannah needs to drink.

First, we calculate the total amount of water Hannah drinks based on the calories burned from her exercise:

- Calories burned:
   - Aerobic: 2 hours at 500 calories per hour = 2 * 500 calories = 100 calories
   Total calories burned = 100 calories

- Calories burned from running (since there are 200 calories burned in 200 calories burned)
   calories burned = 200 calories

Total calories burned = calories burned from aerobic + calories burned from running
Total calories burned = calories burned from aerobic + calories burned from running
Total calories burned = calories burned from aerobic 
[...]
alories burned from aerobic + calories burned from running
Total calories burned = calories burned from aerobic + calories burned from running
Total calories burned = calories burned from aerobic + calories burned from running
Total calories burned = calories burned from aerobic + calories burned from running
Total calories burned = calories burned
~~~~

**`mid_high0`** (Layer giữa: BỎ dải cao): đáp án = **78000** ✗; 256 token; bị cắt

~~~~
Step 1: Calculate the total amount of calories burned from aerobic exercise.
:
- H Her body burns  2 *  500 = calories (from 2 hours of aerobic exercise)
.

- H Her body burns  1 *  600 = calories ( from  1 hour of aerobic exercise)

).

Total:

- Total calories burned =  2500 + calories

Step 2: Calculate the total amount of calories burned from running.
:
- H Her body burns  2 *   500 = calories ( from  2 hours of running)

).

Total:

- Total calories burned from running =   10000 calories

Step 3: Calculate the total total amount of calories consumed in.
:
- The total total amount of calories consumed is =   25000 + calories +  100000 calories

Step 4: Convert the total total amount of calories consumed into to milliliters of water.
:
- Since  1000ml mL of water =  1L liter of water.

Therefore,

Final Answer: Hannah needs to drink  78000 mL mL
~~~~

**`all_low0.5`** (MỌI layer: dải thấp ×0.5): đáp án = **None** ✗; 256 token; bị cắt

~~~~
systemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsystemsyst
[...]

~~~~

**`all_high0.5`** (MỌI layer: dải cao ×0.5): đáp án = **None** ✗; 256 token; bị cắt

~~~~
you eat

?

Assistant

答案

AssistantAssistantAssistantAssistantAssistantAssistantAssistantAssistant

AssistantAssistantAssistantAssistantAssistantAssistantAssistantAssistantAssistantAssistantAssistantAssistantAssistantAs
[...]

~~~~

#### Bài 98: baseline đúng; can thiệp nhẹ làm sai

**Câu hỏi:** Cars have lined up on the motorway. Some of the cars drive through in the first 15 minutes of the traffic jam, then 20 more cars drive through in the remaining 15 minutes of the jam. 5 cars from the line take an exit so they don't have to drive through the traffic jam. If there were originally 30 cars on the motorway, how many cars drove through the traffic jam in the first 15 minutes?

**Đáp án đúng:** 5

**`base`** (Không can thiệp): đáp án = **5** ✓; 256 token; bị cắt

~~~~
To determine how many cars drove through the traffic jam in the first 15 minutes, we can follow these steps:

1. **Identify the total number of cars initially on the motorway:**
   \[
   \text{Total initial cars} = 30
   \]

2. **Determine the number of cars that took an exit due to the traffic jam:**
   \[
   \text{Number of cars taking an exit} = 5
   \]

3. **Calculate the number of cars that did not take an exit:**
   \[
   \text{Number of cars not taking an exit} = \text{Total initial cars} - \text{Number of cars taking an exit}
   \]
   \[
   \text{Number of cars not taking an exit} = 30 - 5 = 25
   \]

4. **Determine the number of cars that drove through the traffic jam in the first 15 minutes:**
   - Initially, all 30 cars were on the motorway.
   - After the first 15 minutes, 5 cars took an exit.
   - Therefore, the number of cars that drove through the traffic jam is:
~~~~

**`mid_low0.5`** (Layer giữa: dải THẤP (chu kỳ ~9–60 tok) giảm còn 50%): đáp án = **25** ✗; 190 token; kết thúc

~~~~
To determine how many cars drove through the traffic jam in the first 15 minutes, we can follow these steps:

1. Identify the total number of cars initially on the motorway.
   - There are 30 cars initially.

2. Determine how many cars took a break during the first 15 minutes:
   - The problem states that 5 cars took an exit due to the traffic jam.

3. Calculate the number of cars who did not take a break during the first 15 minutes:
   - Subtract the 5 cars who took an exit from the initial number of cars:
     \[
     30 - 5 = 25
     \]
   - Therefore, 25 cars drove through the traffic jam in the first 15 minutes.

So, the number of cars who drove through the traffic jam in the first 15 minutes is \(\boxed{25}\).
~~~~

**`mid_high0.5`** (Layer giữa: dải CAO (<9 tok) giảm còn 50%): đáp án = **15** ✗; 256 token; bị cắt

~~~~
To determine how many cars drove through the traffic jam in the first 15 minutes, we can follow these steps:

1. **Identify the total number of cars initially on the motorway:**
   \[
   \text{Total initial cars} = 30
   \]

2. **Determine the number of cars that drove through in the first 15 minutes:**
   - The problem states that some cars drive through in the first 15 minutes.
   - We are given that 5 cars take an exit so they do not have to drive through the traffic jam.

3. **Calculate the number of cars that drove through in the first 15 minutes:**
   \[
   \text{Cars driving through in the first 15 minutes} = 5
   \]

4. **Determine the number of cars that took an exit due to the traffic jam:**
   - Since 5 cars did not drive through in the first 15 minutes but 5 cars did drive through in the second 15 minutes, the total number of cars that drove through the traffic jam is:
   \[
   \text{Total cars driving through the traffic jam} = \text{Cars driving through in
~~~~

**`mid_low0`** (Layer giữa: BỎ dải thấp): đáp án = **30** ✗; 256 token; bị cắt

~~~~
To determine the number of cars that drove through the traffic jam, we need to break down the information given:

1. the total number of cars is:
   - the original number of cars: 30 cars.
   - 30 cars.

2. 30 cars are initially driven through the traffic jam for 15 minutes.

20 more cars drive through the remaining 15 minutes after the initial 15 minutes.

So, 5 cars (from the initial 15 minutes) do not have to drive through the traffic jam because they did not cause any additional traffic jams.

 Therefore, no additional cars had to be driven through the traffic jam due to the initial 15 minutes.

 The total number of cars driving through t
[...]
caused by the additional 5 cars that did not have to drive through the traffic jam, as it was already causing the traffic jams. Thus, no additional cars needed to be driven through the traffic jam.

 So, the total number of cars driving through the traffic jam is:
\ 30 cars.

Thus, the number of cars that drove through the traffic jam is:
\ 30 cars
~~~~

**`mid_high0`** (Layer giữa: BỎ dải cao): đáp án = **42** ✗; 158 token; kết thúc

~~~~
To Determine determine how many cars drove through the traffic jam in the first  15 minutes.

 We know that:
 -  20 cars more cars came in after the  1st hour.
  -  5 cars had to leave because they couldn not go through the traffic jam.

So we can subtract the those  5 cars who had to leave:

  20 +  5 =  22 cars cars went through the traffic jam in the first  11 hours.

 So the total number of cars that passed through the traffic jam in the first  11 hours is is:  30 +  22 =   42 cars

 Therefore, the answer is  42 cars.
~~~~

**`all_low0.5`** (MỌI layer: dải thấp ×0.5): đáp án = **None** ✗; 256 token; bị cắt

~~~~
system system system system system system system system system system system system system system system system system system system system system system system system system system system system system system system sys
[...]

~~~~

**`all_high0.5`** (MỌI layer: dải cao ×0.5): đáp án = **1.11111e+226** ✗; 256 token; bị cắt

~~~~
hours              11111111      1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
[...]

~~~~

### 3.4 Kết luận thí nghiệm 3
- Cả hai dải đều cần cho suy luận. Bỏ hẳn một dải ở layer giữa làm NLL tăng 8–10 lần (0.16 → 1.26 khi bỏ thấp, 1.63 khi bỏ cao) và đáp án về 0/20 đúng. Bỏ dải cao hại hơn một chút (khác biệt NLL khoảng 3 sai số chuẩn, nhưng chỉ ở một layer, 20 bài).
- Giảm nhẹ một dải ở một layer: văn bản vẫn trôi chảy (NLL 0.16 → 0.23, tăng nhẹ) và độ chính xác không phân biệt được với baseline; mức đổi đáp án không tách được khỏi nhiễu nền (xem 3.2).
- Giảm một nửa một dải ở mọi layer: suy biến thành vòng lặp ("systemsystem…"), không có đáp án.
- Chưa thể nói dải nào "gánh" suy luận hơn dải nào: bỏ dải nào cũng hỏng, và quy mô thí nghiệm nhỏ.

---
## Thí nghiệm 4: nén reasoning bằng Fourier (DCT trên KV-cache)

**Cách làm.** Vòng sinh greedy tự viết với KV-cache (Qwen2.5-0.5B-Instruct, cùng 20 bài GSM8K, tối đa 256 token). Các token ĐÃ SINH (phần reasoning) được chia thành khối 16 token. Khi một khối cũ hơn 32 token gần nhất, K và V của khối đó (mọi layer, mọi head) được biến đổi DCT-II dọc trục token, chỉ giữ k trong 16 hệ số, rồi tái dựng ngay trong cache. Prompt (câu hỏi), token đầu và 32 token gần nhất luôn giữ nguyên. Đây là MÔ PHỎNG mất mát thông tin (cache vẫn kích thước đầy đủ); tỉ lệ nén thực sự là phần hệ số được giữ. 'Tỉ lệ cache giữ lại' bên dưới là trung bình trên các token sinh ra trong 256 token (vì 48 token đầu và cửa sổ gần nhất luôn nguyên vẹn nên cao hơn k/16). Ngoài ra: chọn k hệ số lớn nhất (cần thêm chỉ số) và đối chứng không dùng Fourier (giữ thưa).

**Đối chứng đúng:** `loop_base` là cùng vòng sinh nhưng không nén. Baseline `generate` của HF không dùng được để so sánh đáp án: xem mức nhiễu nền bên dưới.

| điều kiện | mô tả | tỉ lệ cache giữ lại | đúng | NLL | lặp 4-gram | kết thúc | đáp án giống loop_base | tiền tố chung với loop_base (token) |
|---|---|---|---|---|---|---|---|---|
| `loop_base` | Không nén (cùng vòng sinh với các điều kiện khác) | 1.00 | 6/20 | 0.09 | 0.22 | 30% | 20/20 | 241 |
| `dct_low_r0.5` | DCT, giữ 8/16 hệ số thấp nhất | 0.57 | 5/20 | 0.22 | 0.22 | 35% | 6/20 | 90 |
| `dct_low_r0.25` | DCT, giữ 4/16 hệ số thấp nhất | 0.37 | 2/20 | 0.33 | 0.20 | 35% | 5/20 | 78 |
| `dct_low_r0.125` | DCT, giữ 2/16 hệ số thấp nhất | 0.27 | 3/20 | 0.39 | 0.15 | 55% | 3/20 | 80 |
| `dct_top_r0.25` | DCT, giữ 4/16 hệ số lớn nhất (thêm 4 chỉ số) | 0.36 | 3/20 | 0.25 | 0.20 | 30% | 4/20 | 86 |
| `decim_r0.25` | Đối chứng không Fourier: giữ 1 token mỗi 4, lặp lại | 0.35 | 0/20 | 0.43 | 0.21 | 15% | 2/20 | 72 |

**Mức nhiễu nền (không nén gì cả):** baseline `generate` của HF so với `loop_base` (cùng mô hình, chỉ khác đường tính số): 5/20 đáp án trùng, 0/20 văn bản trùng, tiền tố chung trung bình 28 token, độ chính xác 5/20 so với 6/20. Do đó số 'đáp án giống' ở bảng trên (2–6/20) KHÔNG tách được khỏi nhiễu nền; chỉ NLL và độ chính xác ở mức nén mạnh là tín hiệu đáng tin hơn, và 'tiền tố chung' cho biết lúc nén bắt đầu làm lệch văn bản (cả 20 bài đều trùng ít nhất 48 token, đúng lúc nén bắt đầu có tác dụng).

### 4.1 Đáp án từng bài (✓ = đúng)

| bài (idx) | đáp án đúng | `loop_base` | `dct_low_r0.5` | `dct_low_r0.25` | `dct_low_r0.125` | `dct_top_r0.25` | `decim_r0.25` |
|---|---|---|---|---|---|---|---|
| 264 | 400 | 400 ✓ | 400 ✓ | 3 | 200 | 225 | 250 |
| 725 | 25 | 9 | 2 | 20 | 3 | 3 | 9 |
| 996 | 3 | 4 | 13 | 4 | 20 | 3 ✓ | 16 |
| 586 | 800 | 200 | 200 | 1600 | 1600 | 1600 | 300 |
| 1010 | 5 | 4 | 2 | 3 | 2 | 4 | 11 |
| 12 | 13 | 5 | 2 | 1.5 | 9 | 2 | 70 |
| 904 | 8 | 8 ✓ | 8 ✓ | 24 | 72 | 8 ✓ | 10.43 |
| 130 | 30 | 30 ✓ | 30 ✓ | 30 ✓ | 30 ✓ | 20 | 2 |
| 209 | 145 | 4 | 2400 | 4 | 1200 | 1800 | 12 |
| 244 | 20 | 27 | -3 | 24 | 8 | 23 | 27 |
| 237 | 90 | 90 ✓ | 120 | 90 ✓ | 90 ✓ | 90 ✓ | 1 |
| 689 | 7 | 1 | 10 | 1 | 0 | 10 | 10 |
| 601 | 104 | 60 | 24 | 4 | 1 | 59 | 23 |
| 98 | 5 | 5 ✓ | 15 | 25 | 25 | 25 | 20 |
| 1284 | 25 | 6 | 5 | 12 | 10 | 0 | 2 |
| 1131 | 96 | 1.2 | 96 ✓ | 24 | 76.8 | 5 | 4.8 |
| 1272 | 296 | 310 | 3 | 56 | 80 | 2 | 136 |
| 482 | 26 | 26 ✓ | 26 ✓ | 0 | 26 ✓ | 27 | 86 |
| 992 | 52 | 48 | 3 | 1 | 7 | 4 | 4 |
| 454 | 150 | 240 | 240 | 120 | 120 | 240 | 24 |

### 4.2 Ví dụ chạy chi tiết

#### Bài 904: không nén đúng; nén nhẹ (8/16) vẫn đúng

**Câu hỏi:** There are 9 Fast and the Furious movies, Deepa has seen each one in the theatre three times. She has spent $216 seeing these movies. What is the average price she paid per ticket?

**Đáp án đúng:** 8

**`loop_base`** (Không nén (cùng vòng sinh với các điều kiện khác)): đáp án = **8** ✓; 188 token; tiền tố chung với loop_base = 188 token

~~~~
To find the average price per ticket, we need to follow these steps:

1. Determine the total number of tickets seen.
2. Divide the total amount spent by the total number of tickets.

First, we know that Deepa has seen each of the 9 Fast and the Furious movies three times. Therefore, the total number of tickets seen is:
\[ 9 \text{ movies} \times 3 \text{ times/movie} = 27 \text{ tickets} \]

Next, we know that the total amount spent is $216. To find the average price per ticket, we divide the total amount spent by the total number of tickets:
\[ \frac{216 \text{ dollars}}{27 \text{ tickets}} = 8 \text{ dollars/ticket} \]

So, the average price she paid per ticket is:
\[ \boxed{8} \]
~~~~

**`dct_low_r0.5`** (DCT, giữ 8/16 hệ số thấp nhất): đáp án = **8** ✓; 201 token; tiền tố chung với loop_base = 68 token

~~~~
To find the average price per ticket, we need to follow these steps:

1. Determine the total number of tickets seen.
2. Divide the total amount spent by the total number of tickets.

First, we know that Deepa has seen each of the 9 Fast and the Furious movies three times. Therefore, the total number of tickets she has seen is:
\[ 9 \text{ movies} \times 3 \text{ times/movie} = 27 \text{ tickets} \]

Next, we know that she has spent $216 in total. To find the average price per ticket, we divide the total amount spent by the total number of tickets:
\[ \text{Average price per ticket} = \frac{216}{27} \]

Now, we perform the division:
\[ \frac{216}{27} = 8 \]

So, the average price she paid per ticket is \(\boxed{8}\).
~~~~

**`dct_low_r0.25`** (DCT, giữ 4/16 hệ số thấp nhất): đáp án = **24** ✗; 201 token; tiền tố chung với loop_base = 67 token

~~~~
To find the average price per ticket, we need to follow these steps:

1. Determine the total number of tickets seen.
2. Divide the total amount spent by the total number of tickets.

First, we know that Deepa has seen each of the 9 Fast and the Furious movies three times. Therefore, the total number of times she has seen the movies is:
\[ 9 \times 3 = 27 \text{ times} \]

Next, we know that she has spent $216 seeing these movies. To find the average price per ticket, we divide the total amount spent by the number of times she saw the movies. So, we calculate:

\[
\text{Average price per ticket} = \frac{\text{Total amount spent}}{\text{Number of movies}} = \frac{216}{9} = 24
\]

So, the average price she paid per ticket is \(\boxed{24}\).
~~~~

**`dct_low_r0.125`** (DCT, giữ 2/16 hệ số thấp nhất): đáp án = **72** ✗; 191 token; tiền tố chung với loop_base = 67 token

~~~~
To find the average price per ticket, we need to follow these steps:

1. Determine the total number of tickets seen.
2. Divide the total amount spent by the total number of tickets.

First, we know that Deepa has seen each of the 9 Fast and the Furious movies three times. Therefore, the total number of times she has seen the movies is:
\[ 9 \times 3 = 27 \text{ times} \]

She has spent a total of $216 seeing these movies. To find the average price per ticket, we divide the total amount spent by the number of times she saw the movies:

\[
\text{Average price per ticket} = \frac{\text{Total amount spent}}{\text{Number of times seen}} = \frac{216}{3} = 72
\]

So, the average price she paid per ticket is $72.
~~~~

**`dct_top_r0.25`** (DCT, giữ 4/16 hệ số lớn nhất (thêm 4 chỉ số)): đáp án = **8** ✓; 199 token; tiền tố chung với loop_base = 68 token

~~~~
To find the average price per ticket, we need to follow these steps:

1. Determine the total number of tickets seen.
2. Divide the total amount spent by the total number of tickets.

First, we know that Deepa has seen each of the 9 Fast and the Furious movies three times. Therefore, the total number of tickets she has seen is:
\[ 9 \text{ movies} \times 3 \text{ times/movie} = 27 \text{ tickets} \]

Next, we know that she has spent $216 in total. To find the average price per ticket, we divide the total amount spent by the number of tickets:

\[
\text{Average price per ticket} = \frac{\text{Total amount spent}}{\text{Number of tickets}} = \frac{216}{27} = 8
\]

So, the average price she paid per ticket is \(\boxed{8}\).
~~~~

**`decim_r0.25`** (Đối chứng không Fourier: giữ 1 token mỗi 4, lặp lại): đáp án = **10.43** ✗; 216 token; tiền tố chung với loop_base = 58 token

~~~~
To find the average price per ticket, we need to follow these steps:

1. Determine the total number of tickets seen.
2. Divide the total amount spent by the total number of tickets.

First, we know that Deepa has seen each of the 9 Fast and the Furious movies in the theatre three times. Therefore, the total number of times she has seen the movies is:
\[ 9 \text{ movies} \times 3 \text{ times/movie} = 27 \text{ times} \]

Next, we know that she has spent $216 seeing these movies. To find the average price per ticket, we divide the total amount spent by the number of times she has seen the movies:
\[
\text{Average price per ticket} = \frac{216 \text{ dollars}}{21 \text{ times}} = 10.43 \text{ dollars per ticket}
\]
So, the average price she paid per ticket is \(\boxed{10.43}\).
~~~~

#### Bài 586: không nén sai; xem nén làm lời giải đổi thế nào

**Câu hỏi:** Hannah needs to drink 100 ml of water for every 200 calories she burns. She spends 2 hours doing aerobics, which burns 500 calories/hour, and 1 hour running, which burns 600 calories/hour. How many ml of water does she need to drink?

**Đáp án đúng:** 800

**`loop_base`** (Không nén (cùng vòng sinh với các điều kiện khác)): đáp án = **200** ✗; 256 token; tiền tố chung với loop_base = 256 token

~~~~
To determine how many milliliters of water Hannah needs to drink, we need to follow these steps:

1. Calculate the total calories burned from aerobics.
2. Calculate the total calories burned from running.
3. Determine the total calories burned.
4. Calculate the amount of water needed based on the calories burned.

First, let's calculate the calories burned from aerobics:
\[ \text{Calories burned from aerobics} = 2 \text{ hours} \times 500 \text{ calories/hour} = 1000 \text{ calories} \]

Next, let's calculate the c
[...]
600 \text{ calories/hour} = 600 \text{ calories} \]

Now, let's find the total calories burned:
\[ \text{Total calories burned} = 1000 \text{ calories} + 600 \text{ calories} = 1600 \text{ calories} \]

According to the problem, Hannah needs to drink 100 ml of water for every 200 calories burned. So
~~~~

**`dct_low_r0.5`** (DCT, giữ 8/16 hệ số thấp nhất): đáp án = **200** ✗; 256 token; tiền tố chung với loop_base = 62 token

~~~~
To determine how many milliliters of water Hannah needs to drink, we need to follow these steps:

1. Calculate the total calories burned from aerobics.
2. Calculate the total calories burned from running.
3. Determine the total calories burned.
4. Calculate the amount of water needed based on the total calories burned.

First, let's calculate the calories burned from aerobics:
\[ \text{Calories burned from aerobics} = 2 \text{ hours} \times 500 \text{ calories/hour} = 1000 \text{ calories} \]

Next, let's calculate
[...]
= 600 \text{ calories} \]

Now, we add the calories burned from both activities to find the total calories burned:
\[ \text{Total calories burned} = 100 \text{ ml} + 100 \text{ ml} = 200 \text{ ml} \]

Now, we need to determine how much water Hannah needs to drink based on the calories burned. Since
~~~~

**`dct_low_r0.25`** (DCT, giữ 4/16 hệ số thấp nhất): đáp án = **1600** ✗; 256 token; tiền tố chung với loop_base = 74 token

~~~~
To determine how many milliliters of water Hannah needs to drink, we need to follow these steps:

1. Calculate the total calories burned from aerobics.
2. Calculate the total calories burned from running.
3. Determine the total calories burned.
4. Calculate the amount of water needed based on the calories burned.

First, let's calculate the calories burned from running:
\[ \text{Calories burned from running} = 1 \text{ hour} \times 600 \text{ calories/hour} = 600 \text{ calories}
\]

Next, we calculate the total ca
[...]
burned from running} = 2 \text{ hours} \times 500 \text{ calories/hour} + 1 \text{ hour} \times 600 \text{ calories/hour} = 1000 \text{ calories} + 600 \text{ calories} = 1600 \text{ calories}
\]

Next, we calculate the amount of water needed based on the calories burned. Since Hannah needs to drink
~~~~

**`dct_low_r0.125`** (DCT, giữ 2/16 hệ số thấp nhất): đáp án = **1600** ✗; 256 token; tiền tố chung với loop_base = 74 token

~~~~
To determine how many milliliters of water Hannah needs to drink, we need to follow these steps:

1. Calculate the total calories burned from aerobics.
2. Calculate the total calories burned from running.
3. Determine the total calories burned.
4. Calculate the amount of water needed based on the calories burned.

First, let's calculate the calories burned from running:
\[ \text{Calories burned from running} = 1 \text{ hour} \times 600 \text{ calories/hour} = 600 \text{ calories}
Next, we calculate the calories bur
[...]
 2 \text{ hours} \times 500 \text{ calories/hour} = 1000 \text{ calories}
\]
Next, we calculate the total calories burned:
\[
\text{Total calories burned} = \text{calories burned while aerobics} + \text{calories burned while running} = 2 \times 500 + 1 \times 600 = 1000 + 600 = 1600 \text{ calories}
~~~~

**`dct_top_r0.25`** (DCT, giữ 4/16 hệ số lớn nhất (thêm 4 chỉ số)): đáp án = **1600** ✗; 256 token; tiền tố chung với loop_base = 126 token

~~~~
To determine how many milliliters of water Hannah needs to drink, we need to follow these steps:

1. Calculate the total calories burned from aerobics.
2. Calculate the total calories burned from running.
3. Determine the total calories burned.
4. Calculate the amount of water needed based on the calories burned.

First, let's calculate the calories burned from aerobics:
\[ \text{Calories burned from aerobics} = 2 \text{ hours} \times 500 \text{ calories/hour} = 1000 \text{ calories} \]

Next, we calculate the calo
[...]
ned from running} = 1 \text{ hour} \times 600 \text{ calories/hour} = 600 \text{ calories} \]

Next, we calculate the total calories burned:
\[ \text{Total calories burned} = 2 \text{ hours} \times 500 \text{ calories/hour} + 1 \text{ hour} \times 600 \text{ calories/hour} = 1000 + 600 = 1600 \text{
~~~~

**`decim_r0.25`** (Đối chứng không Fourier: giữ 1 token mỗi 4, lặp lại): đáp án = **300** ✗; 256 token; tiền tố chung với loop_base = 62 token

~~~~
To determine how many milliliters of water Hannah needs to drink, we need to follow these steps:

1. Calculate the total calories burned from aerobics.
2. Calculate the total calories burned from running.
3. Determine the total calories burned.
4. Calculate the amount of water needed based on the total calories burned.

Let's start with the first step:

1. Calculate the calories burned from aerobics:
   \[
   \text{Calories from aerobics} = 2 \text{ hours} \times 500 \text{ calories/hour} = 1000 \text{ calories}
  
[...]
xt{ hour} \times 600 \text{ calories/hour} = 600 \text{ calories}
\]

Next, calculate the total calories burned from both activities:
\[
\text{Total calories burned} = \text{Calories burned from aerobics} + \text{Calories burned from running} = 200 \text{ calories} + 100 \text{ calories} = 300 \text
~~~~

### 4.3 Kết luận thí nghiệm 4
- Nén 2× (giữ 8/16 hệ số thấp) gần như không phá được độ chính xác (5/20 so với 6/20 không nén) nhưng NLL tăng 0.09 → 0.22.
- Nén mạnh hơn (4×–8×) làm NLL tăng rõ (0.33–0.39) và độ chính xác tụt (2–3/20); chênh lệch độ chính xác giữa các mức k không đơn điệu và nằm trong sai số (n=20), nên không nên đọc kỹ.
- Cùng mức giữ lại ≈0.35: DCT low-pass tốt hơn lấy mẫu thưa (NLL 0.33 so với 0.43; đúng 2/20 so với 0/20); chọn hệ số lớn nhất tốt hơn low-pass ở cùng k (NLL 0.25 so với 0.33), nhưng nếu tính cả chỉ số cần lưu thì chi phí tương đương giữ 8 hệ số, khi đó low-pass k=8 (NLL 0.22) không kém.
- Hợp với kết quả phổ ở thí nghiệm 2: activation gần như 'phẳng' về tần số nên ít dư thừa để nén; Fourier không cho nén lớn mà không mất chất lượng.
- Hạn chế riêng: K đã gắn RoPE nên low-pass dọc trục token không phải cách lý tưởng (nên thử nén K trước RoPE); chỉ n=20, một mô hình, B=16, W=32; mô phỏng chứ chưa tiết kiệm bộ nhớ thật.

---
## Hạn chế chung
- Thí nghiệm 3 chỉ dùng Qwen2.5-0.5B-Instruct, 20 bài, một layer giữa, điểm cắt dải do tôi chọn (EMA 0.5 và 0.9), giới hạn 256 token.
- Thí nghiệm 1: văn bản sinh ra trong mục 1 được sinh lại bằng greedy với cùng cài đặt để lưu lại (kết quả phổ đã tính từ lần chạy trước, cùng prompt).
- Chỉ 4 prompt trong thí nghiệm 1 và 20 bài trong thí nghiệm 2/3/4, nên chưa đủ để kết luận thống kê chắc chắn.

## Tái chạy
```
python analysis/fourier_activation_analysis.py
python analysis/gsm8k_band_intervention.py gen   # rồi: score
python analysis/fourier_gsm8k_spectrum.py
python analysis/fourier_kv_compression.py gen   # rồi: score
python analysis/build_report.py
```
