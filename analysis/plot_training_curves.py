#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vẽ Hình 4.1–4.3 (đường cong huấn luyện) cho khóa luận từ logs/<model>.csv.

  Hình 4.1: hàm mất mát theo epoch (base vs mức 2)
  Hình 4.2: MRR trên tập kiểm định theo epoch (base vs mức 2)
  Hình 4.3: hệ số hợp nhất γ theo epoch (mức 2)

Tự tìm log ICEWS14 base (pathFalse) và mức 2 (pathTrue) mới nhất, hoặc truyền tay.

Cách dùng:
  python analysis/plot_training_curves.py            # tự tìm ICEWS14
  python analysis/plot_training_curves.py --dataset ICEWS18
  python analysis/plot_training_curves.py --base logs/A.csv --m2 logs/B.csv --out figures
"""
import argparse, csv, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_BASE = "#4C72B0"   # xanh
C_M2   = "#C44E52"   # đỏ


def newest(pattern):
    fs = glob.glob(pattern)
    return max(fs, key=os.path.getmtime) if fs else None


def read_log(path):
    """Đọc CSV log -> dict cột -> list (bỏ ô rỗng cho val_*)."""
    epochs, loss, vmrr, vmrr_ep, gamma = [], [], [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            ep = int(r["epoch"])
            epochs.append(ep)
            loss.append(float(r["loss"]))
            gamma.append(float(r["path_gamma"]))
            if r.get("val_mrr_filter", "") not in ("", None):
                vmrr.append(float(r["val_mrr_filter"]) * 100)  # -> %
                vmrr_ep.append(ep)
    return dict(epoch=epochs, loss=loss, vmrr=vmrr, vmrr_ep=vmrr_ep, gamma=gamma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ICEWS14")
    ap.add_argument("--base", default=None)
    ap.add_argument("--m2", default=None)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    base_f = args.base or newest(f"logs/*{args.dataset}*pathFalse*.csv")
    m2_f   = args.m2   or newest(f"logs/*{args.dataset}*pathTrue*.csv")
    if not base_f or not m2_f:
        raise SystemExit(f"Thiếu log: base={base_f}, m2={m2_f}. Cần cả pathFalse và pathTrue.")
    print(f"base = {base_f}\nm2   = {m2_f}")
    os.makedirs(args.out, exist_ok=True)

    B = read_log(base_f)
    M = read_log(m2_f)

    # ---------- Hình 4.1: Loss ----------
    plt.figure(figsize=(7, 4.2))
    plt.plot(B["epoch"], B["loss"], "-o", color=C_BASE, ms=4, label="Baseline (LogCL)")
    plt.plot(M["epoch"], M["loss"], "-s", color=C_M2,   ms=4, label="LogCL-PathHead")
    plt.xlabel("Epoch"); plt.ylabel("Hàm mất mát")
    plt.title(f"Hàm mất mát huấn luyện theo epoch ({args.dataset})")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); p1 = f"{args.out}/hinh_4_1_loss_{args.dataset}.png"
    plt.savefig(p1, dpi=200, bbox_inches="tight"); plt.close()

    # ---------- Hình 4.2: MRR valid ----------
    plt.figure(figsize=(7, 4.2))
    plt.plot(B["vmrr_ep"], B["vmrr"], "-o", color=C_BASE, ms=4, label="Baseline (LogCL)")
    plt.plot(M["vmrr_ep"], M["vmrr"], "-s", color=C_M2,   ms=4, label="LogCL-PathHead")
    # đánh dấu điểm tốt nhất
    for D, c in [(B, C_BASE), (M, C_M2)]:
        if D["vmrr"]:
            i = max(range(len(D["vmrr"])), key=lambda k: D["vmrr"][k])
            plt.scatter([D["vmrr_ep"][i]], [D["vmrr"][i]], s=90, facecolors="none",
                        edgecolors=c, linewidths=1.6, zorder=5)
    plt.xlabel("Epoch"); plt.ylabel("MRR trên tập kiểm định (%)")
    plt.title(f"MRR trên tập kiểm định theo epoch ({args.dataset})")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); p2 = f"{args.out}/hinh_4_2_mrr_{args.dataset}.png"
    plt.savefig(p2, dpi=200, bbox_inches="tight"); plt.close()

    # ---------- Hình 4.3: gamma ----------
    plt.figure(figsize=(7, 4.2))
    plt.axhline(0, color="gray", lw=0.8, ls="--")
    # Thêm điểm khởi tạo γ=0 (log ghi sau epoch 0 nên không có giá trị init thật)
    e0 = M["epoch"][0]
    gx = [e0 - 1] + M["epoch"]
    gy = [0.0] + M["gamma"]
    plt.plot(gx, gy, "-s", color=C_M2, ms=4, label="γ (LogCL-PathHead)")
    plt.scatter([e0 - 1], [0.0], s=70, color="gray", zorder=5)
    plt.annotate("khởi tạo γ = 0", xy=(e0 - 1, 0.0),
                 xytext=(e0 + max(M["epoch"]) * 0.12, 0.045),
                 arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9, color="dimgray")
    plt.xlabel("Epoch"); plt.ylabel("Hệ số hợp nhất γ")
    plt.title(f"Hệ số hợp nhất γ học được theo epoch ({args.dataset})")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); p3 = f"{args.out}/hinh_4_3_gamma_{args.dataset}.png"
    plt.savefig(p3, dpi=200, bbox_inches="tight"); plt.close()

    print("Đã xuất:")
    for p in (p1, p2, p3):
        print(" ", p)


if __name__ == "__main__":
    main()
