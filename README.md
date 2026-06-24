# LogCL
The code of LogCL

## Environment Setup

Yêu cầu môi trường:

```txt
torch==1.12.0
dgl-cu113==1.0.2
tqdm
pandas
rdflib
```

Khuyến nghị sử dụng **Python 3.9** (DGL 1.0.2 cu113 hỗ trợ py36–py310).

### 1. Tạo và kích hoạt môi trường

```bash
conda create -n logcl python=3.9
conda activate logcl
```

### 2. Cài PyTorch

Nếu dùng CUDA 11.3:

```bash
pip install torch==1.12.0+cu113 \
--extra-index-url https://download.pytorch.org/whl/cu113
```

(Nếu không cần GPU thì có thể dùng `pip install torch==1.12.0`.)

### 3. Cài DGL CUDA 11.3

Lưu ý: `dgl-cu113==1.0.2` không cài trực tiếp bằng pip. Với Conda, package nằm trong label `cu113`.

```bash
conda install -c dglteam/label/cu113 dgl=1.0.2
```

### 4. Cài các thư viện còn lại

```bash
pip install tqdm pandas rdflib
```

### 5. Kiểm tra cài đặt

```bash
python -c "
import torch, dgl, tqdm, pandas, rdflib, numpy
print('torch  :', torch.__version__)
print('cuda   :', torch.cuda.is_available())
print('dgl    :', dgl.__version__)
print('numpy  :', numpy.__version__)
print('pandas :', pandas.__version__)
print('tqdm   :', tqdm.__version__)
print('rdflib :', rdflib.__version__)
print('=> All libraries OK')
"
```

Kết quả mong đợi:

```txt
torch  : 1.12.0+cu113
cuda   : True
dgl    : 1.0.2+cu113
numpy  : 1.26.4
pandas : 2.3.3
tqdm   : 4.67.1
rdflib : 7.6.0
=> All libraries OK
```

### 6. Xử lý lỗi thường gặp (Troubleshooting)

Trên các bản kernel mới (vd. Fedora kernel 6.x/7.x) và môi trường thiếu CUDA 11 runtime, bước kiểm tra ở trên có thể báo lỗi dù đã cài đủ thư viện. Dưới đây là 3 lỗi hay gặp và cách khắc phục.

#### 6.1. `libtorch_cpu.so: cannot enable executable stack as shared object requires`

Kernel mới chặn các shared library có cờ `GNU_STACK = RWE` (executable stack). File `libtorch_cpu.so` của PyTorch 1.12 dính cờ này. Xoá cờ executable bằng cách patch ELF (đổi `RWE → RW`):

```bash
LIB="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib/libtorch_cpu.so"
cp -n "$LIB" "$LIB.bak"   # backup
python - "$LIB" <<'EOF'
import sys, struct
path = sys.argv[1]
with open(path, 'r+b') as f:
    data = f.read()
    e_phoff = struct.unpack_from('<Q', data, 0x20)[0]
    e_phentsize = struct.unpack_from('<H', data, 0x36)[0]
    e_phnum = struct.unpack_from('<H', data, 0x38)[0]
    PT_GNU_STACK = 0x6474e551
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        if struct.unpack_from('<I', data, off)[0] == PT_GNU_STACK:
            fo = off + 4
            flags = struct.unpack_from('<I', data, fo)[0]
            f.seek(fo); f.write(struct.pack('<I', flags & ~0x1))  # clear PF_X
            print("PT_GNU_STACK:", flags, "->", flags & ~0x1)
EOF
# Kiểm tra: dòng GNU_STACK phải là RW (không còn E)
readelf -lW "$LIB" | grep GNU_STACK
```

> Có thể thử `execstack -c "$LIB"` trước, nhưng bản execstack cũ thường lỗi `section file offsets not monotonically increasing` với file lớn → dùng cách patch ELF ở trên.

#### 6.2. `libcusparse.so.11: cannot open shared object file`

DGL bản `cu113` cần các CUDA 11 runtime (`libcusparse.so.11`, `libcurand.so.10`...) mà wheel `torch==1.12.0+cu113` không kèm. Cài bộ CUDA 11.3 runtime vào env:

```bash
conda install -c conda-forge cudatoolkit=11.3 -y
```

Sau đó cho DGL tìm thấy các lib này bằng cách thêm `$CONDA_PREFIX/lib` vào `LD_LIBRARY_PATH` **tự động mỗi khi activate env**:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" <<'EOF'
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
EOF
# Activate lại để áp dụng
conda deactivate && conda activate logcl
```

#### 6.3. `NumPy 1.x cannot be run in NumPy 2.0.x`

PyTorch 1.12 được build cho NumPy 1.x. Nếu env có NumPy 2.x, hãy hạ xuống:

```bash
pip install "numpy<2"
```

#### 6.4. Lỗi khi chạy train/test (thiếu file hoặc thư mục)

Khi chạy `python src/main.py ...` có thể gặp các lỗi sau do repo thiếu file/thư mục:

- **`ModuleNotFoundError: No module named 'src.hyperparameter_range'`**
  Thiếu file `src/hyperparameter_range.py`. Tạo lại với nội dung:

  ```python
  """
   Hyperparameter range specification.
  """

  hp_range = {
      "n_hidden": [100, 200, 300, 400],
      "n_layers": [2],
      "dropout": [0.2, 0.4],
      "n_bases": [100],
  }
  ```

- **`FileNotFoundError: ... './models/....pt'`** (khi lưu checkpoint) và
  **`FileNotFoundError: ... './result/<dataset>.csv'`** (khi ghi kết quả test)
  Do thiếu thư mục đích. Code đã được vá để tự tạo thư mục (`os.makedirs(..., exist_ok=True)`), nhưng nếu dùng bản cũ thì tạo tay trước khi chạy:

  ```bash
  mkdir -p models result
  ```

> Lưu ý: code dùng đường dẫn tương đối (`./models`, `./result`, `./data/...`) nên phải chạy từ thư mục gốc của repo.



## Process data
First, unpack the data files 

For the three ICEWS datasets 'ICEWS18', 'ICEWS14', 'ICEWS05-15', and 'GDELT', go into the dataset folder in the `./data` directory and run the following command to construct the static graph and the query historical subgraph.
```
cd ./data/
python get_his_subg.py
cd ./<dataset>
python ent2word.py
cd .. 
python get_his_subg.py
```

## Train models

For each dataset we train **two configurations**, each on **3 seeds** (`123 42 2023`):

- **Baseline** — LogCL with contrastive learning (`--use-cl`), **no** Path Head.
- **Level 2** — same as baseline **plus** the two-hop Path Head (`--use-path --path-level 2`).

The two configs are **identical in every flag except `--use-path`**, so any difference
comes purely from the Path Head. Every run dumps per-query ranks into `ranks/`.

### Per-dataset hyperparameters

`--train-history-len` and `--temperature` differ between datasets; everything else is
shared (`--n-hidden 200 --lr 0.001 --n-layers 2`, static graph, etc.).

| Dataset | `--train-history-len` | `--temperature` | Notes |
|---|:---:|:---:|---|
| ICEWS14 | 7 | 0.03 | smallest, fast |
| ICEWS18 | 7 | 0.03 | largest (23K entities) — heaviest for Level 2 |
| ICEWS05-15 | 9 | 0.07 | very long (4017 timestamps) — slow preprocessing & training |
| GDELT | 7 | 0.07 | huge (1.7M facts) — **no static graph** (drop `--add-static-graph`); not in bundled `data.zip` |

> Values taken from the **LogCL paper** (Implementation Details): history lengths
> **7 / 7 / 9** and temperatures **0.03 / 0.03 / 0.07** for ICEWS14 / ICEWS18 / ICEWS05-15.
> (GDELT: length 7, temperature 0.07, **no** static graph.)

> **`--path-batch-size 64`** is tuned for a **16 GB GPU** (e.g. RTX A4000): it only
> chunks the Path Head computation for memory — it does **not** change results, only
> speed/VRAM. Larger = faster until the GPU saturates. Lower it (`32`/`16`/`8`) if you
> hit `CUDA out of memory`; raise it (`128`) only on ≥24 GB GPUs.

Each run writes a checkpoint to `./models/`, appends a metrics row to
`./result/<dataset>.csv`, and a rank CSV to `ranks/`.

The commands below use **seed 123**. For the full 3-seed average, rerun each with
`--seed 42` and `--seed 2023` (change the `--dump-ranks` filename to match), then report
mean ± std across the 3 seeds.

> All commands below assume `data.zip` is already unpacked (see **Process data**).
> Each dataset block is self-contained: run **Preprocess** once, then **Train**.

### 1. ICEWS14

**Preprocess (run once):**
```bash
cd data
sed -i 's/^dataset_list = .*/dataset_list = ["ICEWS14"]/' get_his_subg.py
(cd ICEWS14 && python ent2word.py)   # static graph -> e-w-graph.txt
python get_his_subg.py               # history subgraph -> his_dict/, his_graph_*/
cd ..
```

**Train — Baseline (seed 123):**
```bash
python src/main.py -d ICEWS14 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu 0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all --add-static-graph --use-cl --temperature 0.03 --seed 123 --dump-ranks ranks/ranks_base_ICEWS14_seed123.csv
```

**Train — Level 2 (seed 123):**
```bash
python src/main.py -d ICEWS14 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu 0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all --add-static-graph --use-cl --temperature 0.03 --use-path --path-dim 32 --path-layers 2 --path-batch-size 64 --path-level 2 --seed 123 --dump-ranks ranks/ranks_m2_ICEWS14_seed123.csv
```

### 2. ICEWS18

Largest graph (23K entities) → heaviest for Level 2. `--path-batch-size 64` fits a 16 GB GPU (~14 GB VRAM); drop to `32`/`16` if it OOMs.

**Preprocess (run once):**
```bash
cd data
sed -i 's/^dataset_list = .*/dataset_list = ["ICEWS18"]/' get_his_subg.py
(cd ICEWS18 && python ent2word.py)
python get_his_subg.py
cd ..
```

**Train — Baseline (seed 123):**
```bash
python src/main.py -d ICEWS18 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu 0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all --add-static-graph --use-cl --temperature 0.03 --seed 123 --dump-ranks ranks/ranks_base_ICEWS18_seed123.csv
```

**Train — Level 2 (seed 123):** if `CUDA out of memory`, lower `--path-batch-size` to `32`/`16`.
```bash
python src/main.py -d ICEWS18 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu 0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all --add-static-graph --use-cl --temperature 0.03 --use-path --path-dim 32 --path-layers 2 --path-batch-size 64 --path-level 2 --seed 123 --dump-ranks ranks/ranks_m2_ICEWS18_seed123.csv
```

### 3. ICEWS05-15

Very long timeline → preprocessing and training take much longer.

**Preprocess (run once — slow):**
```bash
cd data
sed -i 's/^dataset_list = .*/dataset_list = ["ICEWS05-15"]/' get_his_subg.py
(cd ICEWS05-15 && python ent2word.py)
python get_his_subg.py
cd ..
```

**Train — Baseline (seed 123):**
```bash
python src/main.py -d ICEWS05-15 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu 0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all --add-static-graph --use-cl --temperature 0.07 --seed 123 --dump-ranks ranks/ranks_base_ICEWS05-15_seed123.csv
```

**Train — Level 2 (seed 123):**
```bash
python src/main.py -d ICEWS05-15 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu 0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all --add-static-graph --use-cl --temperature 0.07 --use-path --path-dim 32 --path-layers 2 --path-batch-size 64 --path-level 2 --seed 123 --dump-ranks ranks/ranks_m2_ICEWS05-15_seed123.csv
```

### 4. GDELT

Huge (1.7M facts, 2975 timestamps) and **does not use the static graph** — drop
`--add-static-graph` and use `--temperature 0.07`. **Not included in the bundled
`data.zip`**: add `data/GDELT/{train,valid,test,stat}.txt` yourself first.

**Preprocess (run once — no static graph → no `ent2word.py`):**
```bash
cd data
sed -i 's/^dataset_list = .*/dataset_list = ["GDELT"]/' get_his_subg.py
python get_his_subg.py
cd ..
```

**Train — Baseline (seed 123):**
```bash
python src/main.py -d GDELT --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu 0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all --use-cl --temperature 0.07 --seed 123 --dump-ranks ranks/ranks_base_GDELT_seed123.csv
```

**Train — Level 2 (seed 123):**
```bash
python src/main.py -d GDELT --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu 0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all --use-cl --temperature 0.07 --use-path --path-dim 32 --path-layers 2 --path-batch-size 64 --path-level 2 --seed 123 --dump-ranks ranks/ranks_m2_GDELT_seed123.csv
```

### Test a saved checkpoint

Testing needs an explicit checkpoint and the **same architecture flags** used for
training (match `--train-history-len` to the dataset; for a **baseline** checkpoint
drop the `--use-path ...` flags).

```bash
python src/main.py -d ICEWS14 --test \
  --checkpoint models/<checkpoint>.pt \
  --gpu 0 --n-hidden 200 --n-layers 2 \
  --decoder convtranse --encoder uvrgcn --layer-norm \
  --pre-weight 0.9 --pre-type all --add-static-graph --use-cl \
  --train-history-len 7 --test-history-len 7 --temperature 0.03 \
  --use-path --path-dim 32 --path-layers 2 --path-batch-size 64 --path-level 2 \
  --seed 123 --dump-ranks ranks/ranks_m2_ICEWS14_test.csv
```

### Analysis

After training, scripts in `analysis/` consume the rank CSVs in `ranks/` (run from repo root):

```bash
# seed stability of one config
python analysis/analyze_stability.py --files ranks/ranks_base_ICEWS14_seed*.csv
# baseline vs Level 2 (McNemar transition)
python analysis/analyze_transition.py --base ranks/ranks_base_ICEWS14_seed123.csv --new ranks/ranks_m2_ICEWS14_seed123.csv
# repetitive vs new facts
python analysis/analyze_seen_unseen.py -d ICEWS14 --base ranks/ranks_base_ICEWS14_seed123.csv --new ranks/ranks_m2_ICEWS14_seed123.csv
```
## Cite
Please cite our paper if you find this code useful for your research.
~~~
@article{chen2023local,
  title={Local-Global History-aware Contrastive Learning for Temporal Knowledge Graph Reasoning},
  author={Chen, Wei and Wan, Huaiyu and Wu, Yuting and Zhao, Shuyuan and Cheng, Jiayaqi and Li, Yuxin and Lin, Youfang},
  journal={arXiv preprint arXiv:2312.01601},
  year={2023}
}
~~~

