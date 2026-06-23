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
Then the following commands can be used to train the proposed models. By default, dev set evaluation results will be printed when training terminates.

We compare two configurations, each trained on **3 seeds** (`123 42 2023`) for a
fair, reproducible comparison:

- **Baseline** — LogCL with contrastive learning (`--use-cl`), **no** Path Head.
- **Level 2** — same as baseline **plus** the two-hop Path Head (`--use-path`,
  `--path-level 2`).

The two configurations are **identical in every flag except `--use-path`**, so
any difference in results comes purely from the Path Head. Each run dumps its
per-query ranks to a seed-specific CSV (`--dump-ranks`).

1. Baseline (no Path Head), 3 seeds:

```bash
for SEED in 123 42 2023; do
  python src/main.py -d ICEWS14 \
    --train-history-len 7 --test-history-len 7 --dilate-len 1 \
    --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop \
    --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 \
    --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all \
    --add-static-graph --use-cl --temperature 0.03 \
    --seed $SEED \
    --dump-ranks ranks_base_seed${SEED}.csv
done
```

2. Level 2 — two-hop Path Head, 3 seeds. `--path-batch-size` controls Path Head
memory usage; use `8` if `16` still exceeds the available GPU memory.

```bash
for SEED in 123 42 2023; do
  python src/main.py -d ICEWS14 \
    --train-history-len 7 --test-history-len 7 --dilate-len 1 \
    --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop \
    --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 \
    --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all \
    --add-static-graph --use-cl --temperature 0.03 \
    --use-path --path-dim 32 --path-layers 2 --path-batch-size 128 --path-level 2 \
    --seed $SEED \
    --dump-ranks ranks_m2_seed${SEED}.csv
done
```

Each run produces a checkpoint under `./models/`, appends one metrics row to
`./result/ICEWS14.csv`, and writes the rank CSV named above. Report the mean and
standard deviation of the metrics across the 3 seeds for each configuration.

3. Test a saved checkpoint. Testing requires an explicit checkpoint path and
the same architecture flags used for training. For a **baseline** checkpoint,
drop the `--use-path ...` flags below.

```bash
python src/main.py -d ICEWS14 --test \
  --checkpoint models/<checkpoint>.pt \
  --gpu 0 --n-hidden 200 --n-layers 2 \
  --decoder convtranse --encoder uvrgcn --layer-norm \
  --pre-weight 0.9 --pre-type all --add-static-graph --use-cl \
  --train-history-len 7 --test-history-len 7 --temperature 0.03 \
  --use-path --path-dim 32 --path-layers 2 --path-batch-size 16 \
  --seed 123
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

