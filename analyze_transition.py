#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bảng chuyển dịch ĐÚNG/SAI giữa baseline và Mức 2 (kiểu McNemar).

Chỉ dùng thư viện chuẩn (không cần pandas).

Cách dùng:
  # 1) Sinh 2 file rank (chạy test với --dump-ranks)
  python src/main.py -d ICEWS14 --test --checkpoint models/<baseline>.pt \
      ...                                  --dump-ranks ranks_base.csv   # baseline: KHÔNG --use-path
  python src/main.py -d ICEWS14 --test --checkpoint models/<muc2>.pt \
      ... --use-path --path-level 2        --dump-ranks ranks_m2.csv

  # 2) Dựng bảng chuyển dịch
  python analyze_transition.py --base ranks_base.csv --new ranks_m2.csv
"""
import argparse
import csv
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='CSV rank của baseline (--dump-ranks)')
    ap.add_argument('--new', required=True, help='CSV rank của model mới (Mức 2)')
    ap.add_argument('--metric', default='rank_filter',
                    choices=['rank_filter', 'rank_raw'], help='dùng rank lọc hay thô')
    ap.add_argument('--dir', default='both',
                    choices=['fwd', 'inv', 'both'], help='chiều dự đoán: object/subject/cả hai')
    ap.add_argument('--ks', default='1,3,10', help='các ngưỡng Hits@k, vd "1,3,10"')
    args = ap.parse_args()

    b = load(args.base, args.metric, args.dir)
    n = load(args.new, args.metric, args.dir)
    common = sorted(set(b) & set(n))
    if not common:
        sys.exit("Không khớp được truy vấn nào — 2 file phải cùng dataset/test set.")

    print("Số truy vấn so khớp: %d  (base=%d, new=%d)" % (len(common), len(b), len(n)))
    if len(common) != len(b) or len(common) != len(n):
        print("CẢNH BÁO: số dòng lệch — có truy vấn không khớp.")

    rb = [b[k] for k in common]
    rn = [n[k] for k in common]
    mrr_b = sum(1.0 / r for r in rb) / len(rb)
    mrr_n = sum(1.0 / r for r in rn) / len(rn)
    print("MRR base = %.4f | MRR new = %.4f | ΔMRR = %+.4f" % (mrr_b, mrr_n, mrr_n - mrr_b))

    for k in [int(x) for x in args.ks.split(',')]:
        a = bb = cc = dd = 0
        for x, y in zip(rb, rn):
            cb, cn = x <= k, y <= k
            if cb and cn:
                a += 1            # cả hai ĐÚNG
            elif cb and not cn:
                bb += 1           # base đúng -> new sai  (HỒI QUY)
            elif not cb and cn:
                cc += 1           # base sai  -> new đúng (ĐƯỢC VÁ)
            else:
                dd += 1           # cả hai SAI
        N = a + bb + cc + dd

        print("\n========== Hits@%d  (ĐÚNG = rank <= %d) ==========" % (k, k))
        print("                    |  NEW ĐÚNG  |  NEW SAI  |")
        print("  BASE ĐÚNG         | %9d | %8d |" % (a, bb))
        print("  BASE SAI          | %9d | %8d |" % (cc, dd))
        print("  " + "-" * 44)
        print("  Base đúng: %d (%.2f%%)   ->   New đúng: %d (%.2f%%)"
              % (a + bb, 100 * (a + bb) / N, a + cc, 100 * (a + cc) / N))
        print("  ĐƯỢC VÁ (sai->đúng): %d  (%.2f%%)" % (cc, 100 * cc / N))
        print("  HỒI QUY (đúng->sai): %d  (%.2f%%)" % (bb, 100 * bb / N))
        print("  Lợi ích RÒNG = vá - hồi quy = %+d  (%+.2f điểm %%)"
              % (cc - bb, 100 * (cc - bb) / N))
        if bb + cc > 0:
            chi2 = (abs(bb - cc) - 1) ** 2 / (bb + cc)   # McNemar có hiệu chỉnh liên tục
            sig = "CÓ ý nghĩa (p<0.05)" if chi2 > 3.84 else "CHƯA có ý nghĩa"
            print("  McNemar χ²(1) = %.3f  -> %s" % (chi2, sig))


if __name__ == '__main__':
    main()
