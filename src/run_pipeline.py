"""Run all preprocessing steps; never modifies the source GDC files under data/."""
from . import build_expression_matrix, build_metadata, extract_brcadx, inspect_gdc, validate_dataset


def main() -> None:
    inspect_gdc.main()
    build_metadata.main()
    build_expression_matrix.main()
    extract_brcadx.main()
    validate_dataset.main()


if __name__ == "__main__":
    main()
