from pathlib import Path

from .fixed_split import create_fixed_split


def main() -> None:
    create_fixed_split(Path("results/balanced_120_tumor/metadata.csv"))
    print("Created results/fixed_split_assignments.csv")


if __name__ == "__main__":
    main()
