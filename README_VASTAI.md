# Chạy LogCL + Path Head trên vast.ai

Cách chạy dự án trên một instance [vast.ai](https://vast.ai).

---

## 1. Cấu hình instance

| Mục | Chọn |
|-----|------|
| **Image** | `pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel` (gõ tay; `-runtime` nhẹ hơn ~3GB) |
| **GPU** | **RTX A4000 16GB** / 3090 / 4090 |
| **Container disk** | **15GB** (ICEWS14/18) · **40GB** (ICEWS05-15) |
| **Launch Mode** | Interactive shell server, **SSH** |

### Tương thích GPU (stack cu113)
| GPU | Chạy được? |
|-----|-----------|
| 30xx, A4000, A100, V100, A6000 | ✅ native |
| **40xx** (4090, 4080, 4060Ti) | ✅ chạy được (PTX JIT) |
| **50xx** (5070Ti, 5080), L4, H100 | ❌ né — cuBLAS/cuDNN 11.3 không hỗ trợ |

---

## 2. Cài đặt (sau khi SSH vào máy)

torch 1.12 đã có sẵn trong image — chỉ cài thêm dgl + deps:
```bash
git clone https://github.com/buitrongtrinh/LogCL-PathHead.git
cd LogCL-PathHead
unzip -q data.zip
conda install -c dglteam/label/cu113 dgl=1.0.2 -y
pip install "numpy<2" tqdm pandas rdflib
python -c "import torch,dgl; print('cuda:', torch.cuda.is_available())"
```
Phải thấy **`cuda: True`** mới đi tiếp. (Dòng "DGL backend... Assuming PyTorch" chỉ là thông báo.)

> Trên vast.ai **không cần** trò `LD_LIBRARY_PATH` như Colab.

---

## 3. Chạy một dataset (ví dụ ICEWS18)

**Dùng `tmux`** để train không chết khi rớt SSH:
```bash
tmux new -s train        # rớt mạng: ssh lại rồi `tmux attach -t train`
                         # chạy ngầm: Ctrl+b rồi d
```

### Tiền xử lý (1 lần)
```bash
cd data
sed -i 's/^dataset_list = .*/dataset_list = ["ICEWS18"]/' get_his_subg.py
(cd ICEWS18 && python ent2word.py)
python get_his_subg.py
cd .. && mkdir -p ranks
```

### Khai báo COMMON (1 lần / shell)
```bash
COMMON="--train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu 0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all --add-static-graph --use-cl --temperature 0.03"
PATH_FLAGS="--use-path --path-dim 32 --path-layers 2 --path-batch-size 64 --path-level 2"
```

### Baseline — từng seed
```bash
python src/main.py -d ICEWS18 $COMMON --seed 123  --dump-ranks ranks/ranks_base_ICEWS18_seed123.csv
python src/main.py -d ICEWS18 $COMMON --seed 42   --dump-ranks ranks/ranks_base_ICEWS18_seed42.csv
python src/main.py -d ICEWS18 $COMMON --seed 2023 --dump-ranks ranks/ranks_base_ICEWS18_seed2023.csv
```

### Mức 2 (Path Head) — từng seed
```bash
python src/main.py -d ICEWS18 $COMMON $PATH_FLAGS --seed 123  --dump-ranks ranks/ranks_m2_ICEWS18_seed123.csv
python src/main.py -d ICEWS18 $COMMON $PATH_FLAGS --seed 42   --dump-ranks ranks/ranks_m2_ICEWS18_seed42.csv
python src/main.py -d ICEWS18 $COMMON $PATH_FLAGS --seed 2023 --dump-ranks ranks/ranks_m2_ICEWS18_seed2023.csv
```

> **Đổi dataset** (tham số theo paper):
> - **ICEWS14** / **ICEWS18**: `--train-history-len 7 --temperature 0.03`
> - **ICEWS05-15**: `--train-history-len 9 --temperature 0.07`
>
> Nhớ **sed lại `get_his_subg.py`** về đúng dataset trước khi preprocess. Chi tiết xem **README.md** mục *Train models*.

---

## 4. `--path-batch-size` (tốc độ Path Head)

Chỉ chia nhỏ tính toán Path Head cho vừa VRAM — **không đổi kết quả**, càng lớn càng nhanh tới khi GPU bão hòa.

| VRAM | Để |
|------|-----|
| 16GB (A4000) | **64** (~14GB) |
| 24GB (3090/4090) | 128 |
| OOM | hạ 32 → 16 → 8 |

Xem VRAM (tab SSH khác): `nvidia-smi`.

---

## 5. Phân tích

```bash
python analysis/analyze_stability.py --files ranks/ranks_base_ICEWS18_seed*.csv
python analysis/analyze_transition.py --base ranks/ranks_base_ICEWS18_seed123.csv --new ranks/ranks_m2_ICEWS18_seed123.csv
python analysis/analyze_seen_unseen.py -d ICEWS18 --base ranks/ranks_base_ICEWS18_seed123.csv --new ranks/ranks_m2_ICEWS18_seed123.csv
```

---

## 6. ⚠️ Tải kết quả về TRƯỚC khi xóa máy

Xóa instance là mất sạch. Trên **máy bạn** (không phải máy thuê):
```bash
scp -i ~/.ssh/id_ed25519 -P <port> -r root@<ip>:~/LogCL-PathHead/ranks ./ranks_ICEWS18
```

---

## Lỗi thường gặp

| Lỗi | Xử lý |
|-----|-------|
| `Permission denied (publickey)` | Dán SSH public key vào **dialog 🔑 của instance**, đợi 20s |
| `cuda: False` / `no NVIDIA driver` | Chưa được cấp GPU — kiểm tra `nvidia-smi` |
| `no kernel image` | GPU quá mới (50xx) — đổi 30xx/40xx/A4000 |
| `CUDA out of memory` | Hạ `--path-batch-size` (64→32→16) |
| `FileNotFoundError his_graph_*` | Chưa preprocess dataset đó (mục 3) |
| Train chết khi rớt SSH | Phải chạy trong `tmux` |
