import argparse
import json
import random
from pathlib import Path
import numpy as np


def load_xy(client_npz: Path):
    with np.load(client_npz, allow_pickle=True) as z:
        if z.files != ["data"]:
            raise ValueError(f"{client_npz} unexpected keys: {z.files} (expected ['data'])")
        d = z["data"]
        if isinstance(d, np.ndarray) and d.dtype == object and d.shape == ():
            d = d.item()
        if not isinstance(d, dict) or "x" not in d or "y" not in d:
            raise ValueError(f"{client_npz} data is not dict with x,y")
        return d["x"], d["y"]

def merge_split(dataset_dir: Path, split: str, picked_ids):
    xs, ys = [], []
    for cid in picked_ids:
        p = dataset_dir / split / f"{cid}.npz"
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        x, y = load_xy(p)
        xs.append(x)
        ys.append(y)

    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    return X, Y


def save_public_npz(out_path: Path, X, Y, compress: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"data": {"x": X, "y": Y}}
    if compress:
        np.savez_compressed(out_path, **payload)
    else:
        np.savez(out_path, **payload)

def exclude_and_renumber(dataset_dir: Path, picked_ids, exclude_tag: str):
    dataset_dir = Path(dataset_dir)
    picked_ids = set(picked_ids)

    excluded_root = dataset_dir / f"excluded_{exclude_tag}"
    for split in ["train", "test"]:
        dst_dir = excluded_root / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        for cid in picked_ids:
            src = dataset_dir / split / f"{cid}.npz"
            if src.exists():
                src.replace(dst_dir / f"{cid}.npz")  # rename/move :contentReference[oaicite:3]{index=3}

    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    remaining = []
    for p in train_dir.glob("*.npz"):
        try:
            old_id = int(p.stem)  # "12.npz" -> 12
        except ValueError:
            continue
        if old_id not in picked_ids:
            # 确保 test 也存在对应文件
            if (test_dir / f"{old_id}.npz").exists():
                remaining.append(old_id)

    remaining = sorted(remaining)

    mapping = {old: new for new, old in enumerate(remaining)}

    tmp_train = train_dir / "_tmp_renumber"
    tmp_test  = test_dir / "_tmp_renumber"
    tmp_train.mkdir(exist_ok=True)
    tmp_test.mkdir(exist_ok=True)

    for old, new in mapping.items():
        (train_dir / f"{old}.npz").replace(tmp_train / f"{new}.npz")
        (test_dir  / f"{old}.npz").replace(tmp_test  / f"{new}.npz")

    for p in tmp_train.glob("*.npz"):
        p.replace(train_dir / p.name)
    for p in tmp_test.glob("*.npz"):
        p.replace(test_dir / p.name)

    tmp_train.rmdir()
    tmp_test.rmdir()

    (excluded_root / "renumber_map.json").write_text(
        json.dumps(mapping, indent=2),
        encoding="utf-8"
    )

    return mapping  # old_id -> new_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="包含 AGNews/Cifar100/FEMNIST/MNIST 的根目录")
    ap.add_argument("--datasets", type=str, default="AGNews,Cifar100,FEMNIST,MNIST",
                    help="逗号分隔的数据集文件夹名")
    ap.add_argument("--num_clients", type=int, default=30)
    ap.add_argument("--pick", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="public_dataset",
                    help="输出目录名（在 root 下创建）")
    ap.add_argument("--compress", action="store_true",
                    help="使用 np.savez_compressed（更省磁盘，但更慢一些）")
    ap.add_argument("--exclude", action="store_true",
                    help="把抽到的客户端从原 train/test 中移动走（剔除）")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = root / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    rng = random.Random(args.seed)
    picked = sorted(rng.sample(list(range(args.num_clients)), args.pick))

    # 记录抽样，保证四个数据集抽的是同一批 client id（对齐）
    (out_root / "picked_clients.json").write_text(
        json.dumps({"seed": args.seed, "picked": picked, "num_clients": args.num_clients}, indent=2),
        encoding="utf-8"
    )

    for name in datasets:
        dataset_dir = root / name
        if not (dataset_dir / "train").exists() or not (dataset_dir / "test").exists():
            raise FileNotFoundError(f"Bad dataset dir (need train/test): {dataset_dir}")

        Xtr, Ytr = merge_split(dataset_dir, "train", picked)
        Xte, Yte = merge_split(dataset_dir, "test", picked)

        # 保存到 public_dataset/<dataset>/{train,test}/public.npz
        save_public_npz(out_root / name / "train" / "public.npz", Xtr, Ytr, args.compress)
        save_public_npz(out_root / name / "test" / "public.npz",  Xte, Yte, args.compress)

        print(f"[OK] {name}: train={len(Ytr)} test={len(Yte)} -> {out_root/name}")

        if args.exclude:
            mapping = exclude_and_renumber(dataset_dir, picked, exclude_tag=args.out)
            print(f"[OK] {name}: renumbered remaining clients to 0..{len(mapping) - 1}")

    print(f"\nDone. Public dataset root: {out_root}")
    print(f"Picked clients: {picked}")


if __name__ == "__main__":
    main()
