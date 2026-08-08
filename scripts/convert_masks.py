from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

INPUT_FOLDER = (
    ROOT / "data" / "raw" / "masks" / "africa_l3"
)

OUTPUT_FOLDER = (
    ROOT / "data" / "processed" / "masks" / "africa_l3"
)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def convert_xyz_mask(path):
    mask = np.loadtxt(
        path,
        usecols=2,
        dtype=np.float32,
    )

    indices = np.flatnonzero(mask == 1).astype(np.int32)

    output_name = path.name.replace(
        ".mask.xyz",
        ".npz"
    )

    output_path = OUTPUT_FOLDER / output_name

    np.savez_compressed(
        output_path,
        indices=indices,
    )

    return len(mask), len(indices)


def main():
    files = sorted(INPUT_FOLDER.glob("*.mask.xyz"))

    print(f"Found {len(files)} masks")

    for i, path in enumerate(files, start=1):
        total, selected = convert_xyz_mask(path)

        print(
            f"[{i}/{len(files)}] {path.name}: "
            f"{total:,} -> {selected:,} cells"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()