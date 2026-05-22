#!/usr/bin/env python3
"""Check overlap and prepare clean train/val split."""

import os, shutil

train_csv = "/gemini/platform/public/aigc/human_guozz2/code/zhangyan/DiffSynth-Studio-LTX/data/training_data_all_0430.csv"
test_csv  = "/gemini/platform/public/aigc/human_guozz2/code/zhangyan/DiffSynth-Studio-LTX/data/training_data_all_0430_test.csv"
out_dir   = "/gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio/data"

os.makedirs(out_dir, exist_ok=True)

def load_video_paths(path):
    paths = set()
    with open(path, "r") as f:
        f.readline()
        for line in f:
            paths.add(line.split(",")[0].strip())
    print(f"{path}: {len(paths)} videos")
    return paths

def load_lines(path):
    with open(path, "r") as f:
        header = f.readline()
        lines = f.readlines()
    return header, lines

print("Loading...")
test_paths = load_video_paths(test_csv)
train_header, train_lines = load_lines(train_csv)
print(f"Train total: {len(train_lines)} lines")

# Filter: remove test videos from train
clean_train = []
removed = 0
for line in train_lines:
    video = line.split(",")[0].strip()
    if video in test_paths:
        removed += 1
    else:
        clean_train.append(line)

print(f"Removed: {removed} overlapping lines")
print(f"Clean train: {len(clean_train)} lines")

# Write clean train
train_out = os.path.join(out_dir, "training_data_all_0509_clean.csv")
with open(train_out, "w") as f:
    f.write(train_header)
    f.writelines(clean_train)
print(f"Written: {train_out}")

# Copy test as-is
test_out = os.path.join(out_dir, "val.csv")
shutil.copy(test_csv, test_out)
print(f"Copied: {test_out}")

# Verify
print("\nFinal check...")
out_train_paths = set()
with open(train_out) as f:
    f.readline()
    for line in f:
        out_train_paths.add(line.split(",")[0].strip())
out_val_paths = set()
with open(test_out) as f:
    f.readline()
    for line in f:
        out_val_paths.add(line.split(",")[0].strip())
overlap = out_train_paths & out_val_paths
print(f"Train: {len(out_train_paths)} | Val: {len(out_val_paths)} | Overlap: {len(overlap)}")
if overlap:
    print("WARNING: still overlapping!")
else:
    print("OK: train and val are disjoint")
