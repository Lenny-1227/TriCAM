"""Recreate the interaction splits used by the TriCAM experiments.

Raw datasets are intentionally not distributed with this repository. Download
them from their original providers and place them under ``codes/data`` before
running this script.
"""

import argparse
import ast
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DATASETS = ("MenClothing", "WomenClothing", "Beauty", "MicroLens")
SPLIT_NAMES = ("train", "val", "test")


def random_user_split(user_items, seed):
    np.random.seed(seed)
    train_json, val_json, test_json = {}, {}, {}

    for user, items in user_items.items():
        if len(items) < 10:
            held_out = np.random.choice(len(items), 2, replace=False)
        else:
            held_out = np.random.choice(
                len(items), int(len(items) * 0.2), replace=False
            )

        test_indices = held_out[: len(held_out) // 2]
        val_indices = held_out[len(held_out) // 2 :]
        train_indices = [
            index for index in range(len(items)) if index not in held_out
        ]

        train_json[user] = [items[index] for index in train_indices]
        val_json[user] = [items[index] for index in val_indices.tolist()]
        test_json[user] = [items[index] for index in test_indices.tolist()]

    return train_json, val_json, test_json


def prepare_clothing(dataset_dir, seed):
    user_items = defaultdict(list)
    for filename in ("train.csv", "test.csv"):
        path = dataset_dir / filename
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                user_items[int(row["userID"])].append(int(row["itemID"]))
    return random_user_split(user_items, seed)


def prepare_beauty(dataset_dir, seed):
    source = dataset_dir / "meta-data" / "reviews_Beauty_5.json.gz"
    records = []

    with gzip.open(source, "rt", encoding="utf-8") as file:
        for line in file:
            records.append(ast.literal_eval(line))

    items = set(record["asin"] for record in records)
    users = set(record["reviewerID"] for record in records)
    item_to_id = {item: index for index, item in enumerate(items)}
    user_to_id = {user: index for index, user in enumerate(users)}
    user_items = defaultdict(list)

    for record in records:
        user_items[user_to_id[record["reviewerID"]]].append(
            item_to_id[record["asin"]]
        )

    return random_user_split(user_items, seed)


def prepare_microlens(dataset_dir):
    splits = [defaultdict(list), defaultdict(list), defaultdict(list)]
    source = dataset_dir / "microlens.inter"

    with source.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            label = int(row["x_label"])
            if label not in (0, 1, 2):
                raise ValueError("Unexpected x_label: {}".format(label))
            splits[label][int(row["userID"])].append(int(row["itemID"]))

    return tuple(dict(split) for split in splits)


def write_splits(dataset_dir, splits):
    output_dir = dataset_dir / "5-core"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, split in zip(SPLIT_NAMES, splits):
        path = output_dir / (name + ".json")
        with path.open("w", encoding="utf-8") as file:
            json.dump(split, file)
        interactions = sum(len(items) for items in split.values())
        print("{}: users={}, interactions={}".format(path, len(split), interactions))


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=script_dir.parent / "data",
        help="Directory containing the downloaded dataset folders.",
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    dataset_dir = args.data_root.resolve() / args.dataset
    if args.dataset in ("MenClothing", "WomenClothing"):
        splits = prepare_clothing(dataset_dir, args.seed)
    elif args.dataset == "Beauty":
        splits = prepare_beauty(dataset_dir, args.seed)
    else:
        splits = prepare_microlens(dataset_dir)

    write_splits(dataset_dir, splits)


if __name__ == "__main__":
    main()
