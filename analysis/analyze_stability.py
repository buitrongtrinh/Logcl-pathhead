#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Đánh giá ĐỘ ỔN ĐỊNH theo seed của MỘT cấu hình (base hoặc mức 2).

Nhận N file rank (cùng cấu hình, khác seed) do --dump-ranks sinh ra rồi in
MRR / Hits@k của TỪNG seed kèm mean +/- std, và kết luận các seed có cho
kết quả GẦN nhau (ổn định) hay XA nhau (phụ thuộc seed).

Chỉ dùng thư viện chuẩn.

Cách dùng:
  python analyze_stability.py --files ranks_base_seed123.csv \
                                      ranks_base_seed42.csv \
                                      ranks_base_seed2023.csv
  # Mức 2:
  python analyze_stability.py --files ranks_m2_seed*.csv      # shell tự bung *
"""
import argparse
import csv
import os
import statistics
import sys


def load(path, metric, direction):
    """Trả về dict {(time,dir,s,r,o): rank}."""
    out = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if direction != 'both' and row['dir'] != direction:
                continue
            key = (row['time'], row['dir'], row['s'], row['r'], row['o'])
            out[key] = int(row[metric])
    return out


def mrr(ranks):
    return sum(1.0 / r for r in ranks) / len(ranks)


def hits_at(ranks, k):
    return sum(1 for r in ranks if r <= k) / len(ranks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', required=True, nargs='+',
                    help='>=2 file rank cùng cấu hình, khác seed (--dump-ranks)')
    ap.add_argument('--metric', default='rank_filter',
                    choices=['rank_filter', 'rank_raw'], help='dùng rank lọc hay thô')
    ap.add_argument('--dir', default='both',
                    choices=['fwd', 'inv', 'both'], help='chiều dự đoán: object/subject/cả hai')
    ap.add_argument('--ks', default='1,3,10', help='các ngưỡng Hits@k, vd "1,3,10"')
    args = ap.parse_args()

    if len(args.files) < 2:
        sys.exit("Cần ít nhất 2 file (2 seed) để đánh giá độ ổn định.")

    names = [os.path.basename(p) for p in args.files]
    dicts = [load(p, args.metric, args.dir) for p in args.files]

    # chỉ giữ truy vấn có ở TẤT CẢ các seed -> so công bằng
    common = sorted(set.intersection(*[set(d) for d in dicts]))
    if not common:
        sys.exit("Không có truy vấn chung giữa các file — phải cùng dataset/test set.")

    print("Số seed: %d | truy vấn chung: %d" % (len(dicts), len(common)))
    if any(len(d) != len(common) for d in dicts):
        print("CẢNH BÁO: số dòng giữa các seed lệch — chỉ tính trên phần giao.")

    ranks_per_seed = [[d[key] for key in common] for d in dicts]
    ks = [int(x) for x in args.ks.split(',')]

    # bảng điểm: từng seed + mean/std
    print("\n  %-10s" % "metric" + "".join("  %12s" % n[:12] for n in names)
          + "  | %8s  %8s  %8s" % ("mean", "std", "max-min"))
    print("  " + "-" * (12 + 14 * len(names) + 32))

    mrr_vals = [mrr(rk) for rk in ranks_per_seed]

    def row(label, values):
        m = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        spread = max(values) - min(values)
        cells = "".join("  %12.4f" % v for v in values)
        print("  %-10s%s  | %8.4f  %8.4f  %8.4f" % (label, cells, m, sd, spread))

    row("MRR", mrr_vals)
    for k in ks:
        row("Hits@%d" % k, [hits_at(rk, k) for rk in ranks_per_seed])

    # kết luận dựa trên độ biến thiên của MRR
    m = statistics.mean(mrr_vals)
    sd = statistics.stdev(mrr_vals) if len(mrr_vals) > 1 else 0.0
    cv = 100.0 * sd / m if m else 0.0
    print("\n  Biến thiên MRR: std=%.4f  (= %.2f%% so với mean)" % (sd, cv))
    if cv < 1.0:
        verdict = "GẦN nhau => RẤT ỔN ĐỊNH, gần như không phụ thuộc seed."
    elif cv < 3.0:
        verdict = "khá gần => KHÁ ỔN ĐỊNH, ít phụ thuộc seed."
    else:
        verdict = "XA nhau => DAO ĐỘNG đáng kể, kết quả PHỤ THUỘC seed."
    print("  => Kết luận: %s" % verdict)
    print("  (ngưỡng 1%/3% chỉ là quy ước tham khảo, không phải chuẩn cứng)")


if __name__ == '__main__':
    main()
