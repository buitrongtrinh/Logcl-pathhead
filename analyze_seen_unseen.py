#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tách test thành 'LẶP LẠI' (đáp án từng xuất hiện trong lịch sử) vs 'MỚI'
(chưa từng), rồi đo MRR/Hits của base vs mức 2 trên từng nhóm.

Trả lời câu hỏi "LogCL sai loại nào?": nếu base yếu hẳn trên nhóm MỚI và Path
Head cải thiện tập trung ở đó, thì lỗi của LogCL chủ yếu là KHÔNG suy luận
được liên kết mới (chỉ dựa lặp lại lịch sử) — đúng động cơ của Path Head.

Định nghĩa: với truy vấn (s,r,o) tại thời điểm t trong test, fact là 'LẶP LẠI'
nếu (s,r,o) đã xuất hiện ở train + valid + các snapshot test trước t; ngược lại
là 'MỚI'. Cả hướng xuôi (đoán o) và ngược (đoán s) của cùng fact dùng chung nhãn.

Cách dùng:
  python analyze_seen_unseen.py -d ICEWS14 \
      --base ranks_base_seed123.csv --new ranks_m2_seed123.csv
"""
import argparse
import csv
import sys
sys.path.append(".")
import numpy as np
from rgcn import utils


def build_labels(dataset):
    """Trả về (label_map, num_rels) với label_map[(time_idx,s,r,o)] = 'seen'|'new'."""
    data = utils.load_data(dataset)
    num_rels = data.num_rels
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    history = set()
    for snap in train_list + valid_list:
        for s, r, o in snap:
            history.add((int(s), int(r), int(o)))

    label_map = {}
    for t_idx, snap in enumerate(test_list):
        facts = [(int(s), int(r), int(o)) for s, r, o in snap]
        for (s, r, o) in facts:
            label_map[(t_idx, s, r, o)] = 'seen' if (s, r, o) in history else 'new'
        for f in facts:
            history.add(f)
    return label_map, num_rels


def load_ranks_with_label(path, label_map, num_rels, metric):
    """Trả về dict {'seen':[ranks], 'new':[ranks], 'unmatched':n}."""
    out = {'seen': [], 'new': []}
    unmatched = 0
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            t = int(row['time']); rank = int(row[metric])
            s, r, o = int(row['s']), int(row['r']), int(row['o'])
            if row['dir'] == 'inv':            # map nguoc ve fact goc (s,r,o)
                s, r, o = o, r - num_rels, s
            lab = label_map.get((t, s, r, o))
            if lab is None:
                unmatched += 1
            else:
                out[lab].append(rank)
    out['unmatched'] = unmatched
    return out


def mrr(rk):
    return sum(1.0 / r for r in rk) / len(rk) if rk else float('nan')


def hits(rk, k):
    return sum(1 for r in rk if r <= k) / len(rk) if rk else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True)
    ap.add_argument('--base', required=True, help='rank CSV cua base')
    ap.add_argument('--new', required=True, help='rank CSV cua muc 2')
    ap.add_argument('--metric', default='rank_filter', choices=['rank_filter', 'rank_raw'])
    args = ap.parse_args()

    label_map, num_rels = build_labels(args.dataset)
    n_seen = sum(1 for v in label_map.values() if v == 'seen')
    n_new = sum(1 for v in label_map.values() if v == 'new')
    print("\nFact test: %d LẶP LẠI (%.1f%%) | %d MỚI (%.1f%%)  [trên mỗi chiều dự đoán]"
          % (n_seen, 100*n_seen/(n_seen+n_new), n_new, 100*n_new/(n_seen+n_new)))

    b = load_ranks_with_label(args.base, label_map, num_rels, args.metric)
    n = load_ranks_with_label(args.new, label_map, num_rels, args.metric)
    if b['unmatched'] or n['unmatched']:
        print("CẢNH BÁO: không khớp được base=%d, new=%d truy vấn" % (b['unmatched'], n['unmatched']))

    for grp, vn in [('seen', 'LẶP LẠI'), ('new', 'MỚI')]:
        rb, rn = b[grp], n[grp]
        print("\n" + "=" * 64)
        print(" NHÓM: %s   (số truy vấn: %d)" % (vn, len(rb)))
        print("=" * 64)
        print("  %-8s %9s %9s %9s %9s" % ("cfg", "MRR", "H@1", "H@3", "H@10"))
        print("  base     %9.4f %9.4f %9.4f %9.4f" % (mrr(rb), hits(rb, 1), hits(rb, 3), hits(rb, 10)))
        print("  mức 2    %9.4f %9.4f %9.4f %9.4f" % (mrr(rn), hits(rn, 1), hits(rn, 3), hits(rn, 10)))
        print("  Δ        %+9.4f %+9.4f %+9.4f %+9.4f"
              % (mrr(rn)-mrr(rb), hits(rn,1)-hits(rb,1), hits(rn,3)-hits(rb,3), hits(rn,10)-hits(rb,10)))


if __name__ == '__main__':
    main()
