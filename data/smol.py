from pathlib import Path
import sys

import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent

def create_subset(dataset: str):
    # get sources
    train_source_dir, val_source_dir = DATA_DIR / f"{dataset}" / "train", DATA_DIR / f"{dataset}" / "val"
    assert train_source_dir.exists(), f"could not find {train_source_dir}"
    assert val_source_dir.exists(), f"could not find {val_source_dir}"

    # for each, create a `smol` version with just the first row group
    train_target_dir, val_target_dir = DATA_DIR / f"smol-{dataset}" / "train", DATA_DIR / f"smol-{dataset}"/ "val"
    train_target_dir.mkdir(parents=True, exist_ok=True), val_target_dir.mkdir(parents=True, exist_ok=True)

    for source_dir, target_dir in zip([train_source_dir, val_source_dir], [train_target_dir, val_target_dir]):
        move_first_group(source_dir, target_dir)


def move_first_group(source_dir: Path, target_dir: Path):
    """Move the first row group from each parquet file in source_dir to target_dir."""
    for file in source_dir.glob('*.parquet'):
        # Open the existing parquet file
        parquet_file = pq.ParquetFile(file)

        # Read the first row group (index 0)
        table = parquet_file.read_row_group(0)

        # Write the table to a new parquet file
        pq.write_table(table, target_dir / file.name)

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "chess"
    create_subset(dataset)
