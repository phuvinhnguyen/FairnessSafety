# FairnessSafety

Pipeline nhỏ gọn để chạy các thí nghiệm rescue-choice bằng vLLM:

`YAML -> data -> experiment function -> vLLM -> one JSONL/model`

## Cài đặt bằng Conda

```bash
source /apps/local/conda_init.sh
conda create -n vinh_vllm python=3.11 -y
conda activate vinh_vllm
pip install -e '.[data,inference]' --extra-index-url https://download.pytorch.org/whl/cu128
```

## Chạy

Một file YAML:

```bash
bash run.sh configs/missing_people.yaml
```

Tất cả YAML trong một thư mục:

```bash
bash run.sh configs
```

Trên SLURM, các resource mặc định được lấy từ `~/example.sh`:

```bash
CONDA_ENV=vinh_vllm sbatch --export=ALL run.sh configs
```

Chạy bộ so sánh bằng một GPU cho mỗi config:

```bash
CONDA_ENV=vinh_vllm sbatch --array=0-7 --export=ALL run.sh configs/experiments
```

Có thể dùng ít GPU hơn, ví dụ `--array=0-3`; mỗi task sẽ nhận nhiều config theo
kiểu round-robin. Thứ tự được lấy từ tên file đã sort, vì vậy prefix `01_`,
`02_`, ... giữ mapping ổn định. Chỉ dùng một range liên tục hoặc có step, ví dụ
`0-7` hay `0-7:2`.

Bộ `configs/experiments/` chạy Qwen2.5 0.5B/3B/7B và Mistral 7B trên cùng
2.000 trials/config. `01` là baseline đã ẩn demographic; `02`–`05` lần lượt
thêm name, gender, race và age; `06` kết hợp demographics nhưng bỏ name; `07`
dùng toàn bộ thông tin; `08` lặp lại `07` với fairness instruction. Vì mọi file
dùng cùng seed, chênh lệch giữa các cặp này là tín hiệu ablation trực tiếp.

`configs/smoke.yaml` dùng Qwen2.5-0.5B-Instruct với 1.024 prompts và
`max_num_seqs: 256`. `chat_template: true` áp dụng chat template của tokenizer;
đặt `false` hoặc bỏ field này cho base model.

`run.sh` tạo marker trong `.run_state/`. YAML đã hoàn tất hoặc đang được process
bởi một invocation khác sẽ được bỏ qua. Muốn chạy lại một YAML, xóa marker `.done`
tương ứng và đổi `output_dir` (hoặc xóa output cũ). Nếu process bị `SIGKILL`, xóa
thủ công marker `.running` bị stale rồi chạy lại.

Lock được tạo bằng `mkdir`, nên nhiều array job hoặc nhiều lệnh `sbatch` được nộp
cùng lúc vẫn chỉ có một process giành được từng config. `configs/experiments/`
dùng cùng seed, số group và permutation để các ablation/model có thể so sánh trực tiếp.

Mỗi model tạo đúng một file `<output_dir>/<model-name>.jsonl`. File chỉ được đổi
từ `.partial` sang `.jsonl` sau khi model hoàn tất. Mỗi dòng chứa prompt, sample,
candidate mapping, raw response, quyết định đã parse và toàn bộ metadata để tái lập.

## Thêm experiment

Thêm một hàm vào `src/fairness_safety/experiments.py`:

```python
@experiment
def my_experiment(samples):
    return "prompt built from samples"
```

Sau đó chỉ cần dùng đúng tên hàm trong YAML:

```yaml
experiment:
  name: my_experiment
```

Các path trong YAML được resolve tương đối với chính file YAML.

## Cấu trúc

```text
configs/                       YAML experiments
src/fairness_safety/data.py    load và normalize data
src/fairness_safety/experiments.py
src/fairness_safety/runner.py  config, vLLM, parse, JSONL
run.sh                         Bash + SLURM launcher
```
