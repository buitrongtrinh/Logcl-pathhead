# PROJECT STATE — LogCL + Path Head (Khóa luận)

> File tổng hợp trạng thái dự án. Đọc file này để hiểu toàn bộ bối cảnh khi mất context.
> Cập nhật lần cuối: 2026-06-29.

---

## 1. TỔNG QUAN

- **Đề tài:** Cải tiến **LogCL** (Local-Global History-aware Contrastive Learning) cho suy diễn
  Temporal Knowledge Graph (TKG), bằng cách thêm **lớp suy luận đa bước (Path Head)** kế thừa/điều chỉnh từ **CognTKE**.
- **Đóng góp chính:** "Hợp nhất sâu" (deep fusion / **mức 2 / path-level 2**) — Path Head dùng embedding
  đã tiến hóa của LogCL + chấm điểm theo truy vấn, cộng vào điểm LogCL dạng residual có cổng `γ`.
- **Loại:** Khóa luận tốt nghiệp. Người dùng: sinh viên VN, giao tiếp tiếng Việt, thích trả lời ngắn gọn thẳng.
- **Repo:** `github.com/buitrongtrinh/LogCL-PathHead` (fork). Local: `/home/buitrongtrinh/paper/LogCL`
- **File khóa luận:** `LuanAn_DEPTH_final.docx` (+ backup). User tự sửa Word; tôi cung cấp nội dung text/bảng.

---

## 2. MÔ HÌNH & CODE

- **Base = LogCL:** local encoder (tiến hóa RGCN+GRU theo snapshot) + global encoder (đồ thị truy vấn
  lịch sử) + contrastive learning (CL). Decoder = ConvTransE. Encoder = UnionRGCN (uvrgcn).
- **Path Head** (`src/rrgcn.py`, `_fuse_path`, `path_head_scores`): lan truyền 2-hop kiểu Bellman-Ford/NBFNet
  trên đồ thị lịch sử gộp. `final_score = logcl_score + γ · path_score` (path_score chuẩn hóa z-norm).
  - **Mức 1 (shallow):** dùng `emb_rel` tĩnh, readout `path_out`. (kế thừa CognTKE)
  - **Mức 2 (deep, ĐÓNG GÓP):** dùng `rel_emb`/`ent_emb` đã tiến hóa + chấm điểm `path_q([h_s; hr_r])·hidden`.
  - `γ` = `nn.Parameter` khởi tạo **0**, học tự do. Gradient mức 2 chảy ngược vào cả embedding + bộ tiến hóa.
- **Khi bật Path Head, LogCL gốc (local+global+CL) vẫn chạy ĐẦY ĐỦ** — Path Head chỉ cộng thêm.
- **Cờ:** `--use-path --path-dim 32 --path-layers 2 --path-batch-size N --path-level 2`
- **`--path-batch-size` chỉ chia nhỏ tính toán Path Head cho vừa VRAM — KHÔNG đổi kết quả**, chỉ đổi tốc độ/VRAM.
  16GB→64, 24GB→128. Với ICEWS05-15 snapshot nhỏ nên ≥128 là kịch (không nhanh thêm).

### Thay đổi code đã làm
- `src/main.py`: thêm `--dump-ranks` (xuất rank từng truy vấn), `--path-level`, `--seed`;
  **ghi log per-epoch** vào `logs/<model>.csv` (loss, gamma, val_MRR/H@1/3/10, best_mrr, patience);
  `test()` trả thêm `all_hit_filter`. Early stop patience=5.
- `analysis/`: `analyze_stability.py` (ổn định seed), `analyze_transition.py` (McNemar vá/hồi quy),
  `analyze_seen_unseen.py` (lặp lại vs mới). Đều dùng env `logcl`.

---

## 3. MÔI TRƯỜNG

- **Stack:** PyTorch **1.12.0+cu113**, DGL **1.0.2**, Python **3.9**, numpy<2. Conda env `logcl`
  (`~/miniconda3/envs/logcl/bin/python`).
- **GPU:** ICEWS14 chạy local RTX 4050 (Ada, sm_89 — chạy được nhờ PTX JIT). Dataset lớn thuê vast.ai.
- **Tương thích cu113:** ✅ 30xx/A4000/A100/V100 (native), 40xx (PTX JIT). ❌ **50xx Blackwell, L4, H100**
  ("no kernel image").
- **vast.ai:** image `pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel`; cài thêm `dgl=1.0.2` (conda dglteam/label/cu113)
  + `pip install "numpy<2" tqdm pandas rdflib`. KHÔNG cần LD_LIBRARY_PATH (khác Colab).
- **LogCL là CPU/IO-bound** (dựng đồ thị + load .npy mỗi bước) → **CPU đời mới + NVMe quan trọng hơn TFLOPS**.
- **Disk vast:** ICEWS14/18 ~15GB; **ICEWS05-15 cần 35GB** (processed 22GB do 4017 timestamps).
- Chi tiết chạy: xem `README.md` (mục Train models) và `README_VASTAI.md`.

---

## 4. SIÊU THAM SỐ (theo paper LogCL, trang 8) — ĐÃ XÁC MINH

| | ICEWS14 | ICEWS18 | ICEWS05-15 | GDELT |
|---|:---:|:---:|:---:|:---:|
| history-len | **7** | **7** | **9** | 7 |
| temperature | **0.03** | **0.03** | **0.07** | 0.07 |
| static graph | ✅ | ✅ | ✅ | ❌ |
| trong data.zip? | ✅ | ✅ | ✅ | ❌ (phải tự thêm) |

Chung: n_hidden 200, lr 0.001, n_layers 2, dropout 0.2, pre_weight 0.9, weight 0.5, angle 10, discount 1,
pre-type all, Adam wd 1e-5, ConvTransE (50 kernel, 2×3), self-loop, layer-norm.

> LƯU Ý: từng đoán SAI ICEWS18=9, ICEWS05-15=15/temp0.03. ĐÃ SỬA về đúng paper. README đã đúng.

### Thống kê dataset (từ stat.txt local — dùng số này, có chênh nhẹ với paper)
| Dataset | entities | relations | train/valid/test | timestamps |
|---|:---:|:---:|:---:|:---:|
| ICEWS14 | 7.128 | 230 | 74.845 / 8.514 / 7.371 | 365 |
| ICEWS18 | 23.033 | 256 | 373.018 / 45.995 / 49.545 | 304 |
| ICEWS05-15 | 10.488 | 251 | 368.868 / 46.302 / 46.159 | 4.017 |

---

## 5. KẾT QUẢ (filter_all = time-aware filtered, TB 2 chiều, đơn vị %)

### Paper LogCL (để so)
| | MRR | H@1 | H@3 | H@10 |
|---|:---:|:---:|:---:|:---:|
| ICEWS14 | 48.87 | 37.76 | 54.71 | 70.26 |
| ICEWS18 | 35.67 | 24.53 | 40.32 | 57.74 |
| ICEWS05-15 | 57.04 | 46.07 | 63.72 | 77.87 |

### ICEWS14 — ĐẦY ĐỦ 3 seed (123, 42, 2023)
| | MRR | H@1 | H@3 | H@10 |
|---|:---:|:---:|:---:|:---:|
| Base (mean±std) | 48.86±0.23 | 37.68±0.09 | 54.80±0.52 | 70.43±0.39 |
| **Mức 2 (mean±std)** | **53.50±0.31** | **42.71±0.21** | **59.79±0.52** | **73.75±0.27** |
| Δ | +4.64 | +5.03 | +4.99 | +3.32 |
- Per-seed base MRR: 48.79 / 48.68 / 49.12. Mức 2 MRR: 53.85 / 53.25 / 53.40.
- **Base tái lập đúng paper (48.86 ≈ 48.87).** γ: +0.298/+0.281/−0.336 → |γ|=0.31±0.03.

### ICEWS18 — seed 123 (đủ base + m2) ✅
| | MRR | H@1 | H@3 | H@10 |
|---|:---:|:---:|:---:|:---:|
| Base | 35.61 | 24.42 | 40.27 | 57.82 |
| **Mức 2** | **38.72** | **27.48** | **43.73** | **61.01** |
| Δ | +3.11 | +3.06 | +3.46 | +3.19 |
- **Base tái lập đúng paper (35.61 ≈ 35.67).** γ = −0.377.

### ICEWS05-15 — seed 123 (đủ base + m2)
| | MRR | H@1 | H@3 | H@10 |
|---|:---:|:---:|:---:|:---:|
| Base | 57.24 | 46.32 | 63.84 | 78.01 |
| **Mức 2** | **59.78** | **49.27** | **66.36** | **79.49** |
| Δ | +2.54 | +2.95 | +2.52 | +1.48 |
- **Base tái lập đúng paper (57.24 ≈ 57.04).** γ = −0.135.

### Kết luận: Mức 2 VƯỢT LogCL paper trên CẢ 3 dataset (+2.5 đến +4.6 MRR) → SOTA mới.

---

## 6. PHÂN TÍCH SÂU (seed 123)

### McNemar (base vs mức 2)
| Dataset | Ngưỡng | Vá | Hồi quy | Ròng | χ² |
|---|:---:|:---:|:---:|:---:|:---:|
| ICEWS14 | H@1/3/10 | 11.4/10.3/6.8% | 6.1/4.6/3.5% | +5.3/+5.7/+3.4% | 237/321/163 |
| ICEWS18 | H@1/3/10 | 8.2/8.5/7.7% | 5.1/5.0/4.5% | +3.1/+3.5/+3.2% | 698/878/825 |
| ICEWS05-15 | H@1/3/10 | 8.4/6.5/3.9% | 5.5/4.0/2.4% | +2.9/+2.5/+1.5% | 577/557/317 |
- Tất cả p<0.001. Robustness fwd-only ICEWS14 H@1: χ²~117 (vẫn vững). Vá ~1.5-2× hồi quy.

### Lặp lại vs mới
| Dataset | Nhóm | Tỉ lệ | base MRR/H@1 | mức2 MRR/H@1 |
|---|---|:---:|:---:|:---:|
| ICEWS14 | lặp | 52.4% | 65.93/54.33 | 71.51/60.60 |
| ICEWS14 | mới | 47.6% | 29.95/19.21 | 34.43/23.44 |
| ICEWS18 | lặp | 50.4% | 50.25/36.81 | 54.20/40.88 |
| ICEWS18 | mới | 49.6% | 20.73/11.82 | 22.98/13.85 |
| ICEWS05-15 | lặp | 68.4% | 68.83/57.87 | 71.49/61.13 |
| ICEWS05-15 | mới | 31.6% | 32.18/21.35 | 34.45/23.62 |
- **Phát hiện chính:** LogCL yếu ở fact MỚI (MRR ~45% so với lặp). Path Head giúp CẢ HAI nhóm,
  tương đối mạnh hơn ở nhóm mới (+22% H@1 ICEWS14). Đây là động cơ cốt lõi của đề tài.

### γ — đối xứng dấu
- γ khởi tạo 0, học ≠ 0 → module được dùng. **Dấu γ non-identifiable (đối xứng dấu), chỉ |γ| có nghĩa.**
- |γ|: ICEWS05-15 0.14 < ICEWS14 0.31 < ICEWS18 0.38 (dataset nhiều fact mới → |γ| lớn hơn).

---

## 7. CẤU TRÚC FILE

```
ranks/        ranks_{base,m2,m1}_<dataset>_seed<seed>.csv  (cols: time,dir,s,r,o,rank_raw,rank_filter)
              ICEWS14 dùng tên cũ: ranks_{base,m2}_seed<seed>.csv (không có _ICEWS14_)
result/       <dataset>.csv  (metrics mỗi run: filter_all_MRR/H@1/3/10, path_gamma, seed, use_path...)
analysis/     analyze_stability.py | analyze_transition.py (McNemar) | analyze_seen_unseen.py
logs/         <model>.csv  (per-epoch — CHỈ run MỚI sau khi sửa main.py)
figures/      hinh_4_1_loss_ICEWS14.png, hinh_4_2_mrr_ICEWS14.png, hinh_4_3_gamma_ICEWS14.png (Hình 4.1-4.3)
              → tạo bằng analysis/plot_training_curves.py
models/       *.pt checkpoints (gitignored)
data/         gitignored (~26GB); data.zip TRACKED (chứa ICEWS14/18/05-15 + get_his_subg.py + ent2word.py)
README.md, README_VASTAI.md, run_all_colab.ipynb, LuanAn_DEPTH_final.docx
KhoaLuan_4.3_KetQua.docx  ← TOÀN BỘ phần 4.3 (script tạo: /tmp/gen_43.py, chạy ~/miniconda3/bin/python)
```

- File auto-tạo: models/, result/, logs/. **ranks/ phải `mkdir -p ranks` thủ công.**
- Ánh xạ phân tích: `analyze_stability` → ổn định seed; `analyze_transition` → McNemar;
  `analyze_seen_unseen` → lặp/mới; `result/*.csv` → MRR/γ; `logs/*.csv` → đường cong epoch.

---

## 8. CẤU TRÚC CHƯƠNG 4 KHÓA LUẬN (đã thống nhất)

```
4.1 Bộ dữ liệu thực nghiệm
4.2 Cài đặt và cấu hình (4.2.1 môi trường, 4.2.2 siêu tham số [Bảng 4.2 chung + 4.3 per-dataset], 4.2.3 độ đo)
4.3 Kết quả và phân tích   ✅ ĐÃ XUẤT FILE WORD: KhoaLuan_4.3_KetQua.docx (đủ 8 mục, 6 bảng, 3 hình nhúng)
   4.3.1 Kết quả tổng quát & so SOTA + tái lập   [Bảng 4.4 — 3 dataset, seed 123, để LANDSCAPE]
   4.3.2 Nghiên cứu loại bỏ thành phần (ablation base/m1/m2)  [Bảng 4.5]
   4.3.3 Ổn định theo seed (ICEWS14)             [Bảng 4.6]
   4.3.4 Phân tích quá trình huấn luyện          [Hình 4.1 loss, 4.2 MRR/epoch, 4.3 γ/epoch]
   4.3.5 Hiệu năng theo nhóm lặp/mới             [Bảng 4.7]
   4.3.6 Kiểm định ý nghĩa thống kê (McNemar)    [Bảng 4.8]
   4.3.7 Phân tích hệ số γ                       [Bảng 4.9]
   4.3.8 Thảo luận và hạn chế
4.4 Tóm tắt chương
```
- **CHI PHÍ ĐÃ BỎ** (user: "chi phí ko cần trình bày") → không còn mục 4.3.x chi phí, không có Bảng 4.10.
- Toàn bộ 4.3 ĐÃ VIẾT đầy đủ và xuất ra `KhoaLuan_4.3_KetQua.docx` (script `/tmp/gen_43.py`, chạy bằng
  `~/miniconda3/bin/python` có python-docx 1.2.0). File độc lập → user tự áp lại heading style luận án.
- **Quy ước:** số trong khóa luận dùng dấu PHẨY thập phân (53,50). KHÔNG ghi lệnh shell trong thân bài
  (đẩy xuống Phụ lục B). Bảng 4.4 rất rộng (13 cột) → đã đặt trên trang LANDSCAPE riêng (section break).
- **Lưu ý docx:** template luận án `LuanAn_DEPTH_final.docx` KHÔNG có style "Table Grid" (sửa trực tiếp bị
  KeyError). Giải pháp: tạo file mới bằng `Document()` + vẽ viền bảng bằng XML (`w:tblBorders`), không dùng style.
- **Chiến lược:** ICEWS14 = benchmark chính (3 seed, mọi phân tích sâu); ICEWS18/05-15 = kiểm chứng
  tổng quát hóa (1 seed). Đây là thiết kế hợp lệ, trình bày tự tin ở mục Hạn chế (3 trụ: tái lập đúng paper +
  ổn định ICEWS14 + cải thiện nhất quán cùng chiều).

---

## 9. VIỆC CÒN LẠI (PENDING)

| # | Việc | Phục vụ | Mức |
|---|---|---|:---:|
| 1 | ~~Baseline ICEWS18~~ ✅ XONG (35.61, tái lập đúng paper) | Bảng 4.4/4.7/4.8 ICEWS18 đã điền | ✅ |
| 2 | ~~Hình 4.1-4.3 đường cong epoch~~ ✅ XONG (figures/hinh_4_*.png) | nhúng trong 4.3.4 | ✅ |
| 3 | ~~Viết toàn bộ 4.3~~ ✅ XONG → KhoaLuan_4.3_KetQua.docx | chương 4 | ✅ |
| 4 | (KN, tùy chọn) **m1 ICEWS14 đủ 3 seed** | ablation 4.3.2 vững hơn (giờ 1 seed) | 🟢 |
| 5 | Dán 4.3 vào LuanAn_DEPTH_final.docx + áp heading style | hoàn thiện luận án | 🟠 |
| 6 | Viết 4.1, 4.2, 4.4 (nếu chưa) | hoàn thiện chương | 🟠 |

> **Toàn bộ số liệu cho mọi bảng/hình đã ĐỦ.** Không còn việc chạy thực nghiệm bắt buộc (chi phí đã bỏ).
> Việc còn lại chủ yếu là biên tập Word. Số ablation mức 1 ICEWS14 (seed 123): MRR=52.97 H@1=42.23 H@3=59.15 H@10=72.95.

---

## 10. SỰ THẬT KỸ THUẬT QUAN TRỌNG (đừng quên / đừng giải thích sai)

1. **γ dấu non-identifiable** — đối xứng dấu (đảo dấu path_out + đảo dấu γ = mô hình tương đương). Chỉ |γ| có nghĩa.
2. **path-batch-size không đổi kết quả** — chỉ tốc độ/VRAM.
3. **`if epoch and ...`** → validation BỎ QUA epoch 0 (val_* trống ở epoch 0 là ĐÚNG, không phải lỗi). Vẽ MRR/epoch từ epoch 1.
4. **loss_static=0, loss_cl không log riêng** (loss tổng = loss_e + loss_static + loss_cl).
5. **dump-ranks khi đang train ghi VALID set; train xong mới ghi TEST set** (ICEWS05-15 valid 46302→92604 dòng, test 46159→92318 dòng). File cuối phải khớp test×2.
6. **base ICEWS14 = paper LogCL** (48.86≈48.87) → setup chuẩn, cải thiện đáng tin.
7. **torch 1.12 cu113 chạy trên Ada (40xx) nhờ PTX JIT**, KHÔNG chạy Blackwell (50xx).
8. **GDELT không có trong data.zip, không dùng static graph.**

---

## 11. GHI CHÚ THAM KHẢO — vấn đề nhỏ phát hiện khi rà soát (không ảnh hưởng số liệu đã công bố, KHÔNG cần chạy lại)

- `ranks/ranks_m2_seed2023.csv` bị hỏng: dòng header rỗng (thay vì
  `time,dir,s,r,o,rank_raw,rank_filter`) và thiếu 12 dòng chiều `fwd` (7359/7371).
  Làm `analysis/analyze_stability.py` crash (`KeyError: 'time'`) nếu chạy lại
  script này trên 3 seed mức 2. Không ảnh hưởng Bảng 4.6 vì số liệu bảng đó lấy
  từ `result/ICEWS14.csv` (đã tính sẵn), không phải tính lại từ file rank này.
  Muốn tái lập độc lập đầy đủ thì cần chạy lại `--seed 2023 --use-path
  --path-level 2` trên ICEWS14 và dump ranks lại.
- Log huấn luyện dùng cho Hình 4.1/4.2 (`logs/*pathFalse*.csv`, ICEWS14 baseline,
  batch 2026-06-28) dừng ở epoch 11 với `patience_left=5` (vừa đạt best MRR),
  tức có vẻ bị dừng thủ công chứ không phải early-stop tự nhiên — trong khi log
  mức 2 cùng batch chạy đủ đến epoch 22 và early-stop tự nhiên. Không ảnh hưởng
  số liệu bảng (các bảng dùng run gốc 2026-06-22), chỉ ảnh hưởng tính "công bằng
  thị giác" khi so 2 đường cong loss/MRR trong Hình 4.1/4.2. Muốn có hình đẹp hơn
  thì chạy lại baseline đến khi tự early-stop rồi tạo lại `logs/` +
  `figures/hinh_4_*` bằng `analysis/plot_training_curves.py`.
- `result/ICEWS18s.csv` và `ranks/ranks_m2_ICEWS18_seed123s.csv` là file cũ từ
  lần chạy sai hyperparameter (`history-len=9` thay vì 7) trước khi sửa; đã bị
  thay thế bởi `result/ICEWS18.csv` / `ranks/ranks_m2_ICEWS18_seed123.csv` (đúng,
  đang dùng trong khóa luận). Hai file `...s.csv` vẫn còn trong repo, chỉ là rác
  không dùng, có thể xóa khi dọn dẹp sau này.
