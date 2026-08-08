from src.data import load_grace, load_all_masks


def main():
    grace = load_grace()
    masks = load_all_masks()

    print("\n--- TEST ---")

    print("GRACE shape:")
    print(grace["lwe_thickness"].shape)

    first_basin = next(iter(masks))

    print("\nFirst basin:")
    print(first_basin)

    print("Number of cells:")
    print(len(masks[first_basin]))

    print("\nFirst 10 grid indices:")
    print(masks[first_basin][:10])


if __name__ == "__main__":
    main()