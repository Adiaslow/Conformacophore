# src/scripts/convert_xtc.py

import argparse
import logging
from pathlib import Path
from src.core.services.xtc_converter import XTCConverter


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration.

    Args:
        verbose: Whether to enable verbose logging

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("convert_xtc")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    # Create console handler
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    return logger


def main():
    """Main function to convert XTC trajectories to PDB format."""
    parser = argparse.ArgumentParser(
        description="Convert XTC trajectories to PDB format"
    )
    parser.add_argument("xtc_path", help="Path to XTC trajectory file")
    parser.add_argument("top_path", help="Path to topology file")
    parser.add_argument("output_path", help="Path for output PDB file")
    parser.add_argument("--compound-name", help="Name of the compound", required=True)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()
    logger = setup_logging(args.verbose)

    try:
        # Create converter instance
        converter = XTCConverter(verbose=args.verbose)

        # Convert trajectory
        converter.convert(
            xtc_path=args.xtc_path,
            top_path=args.top_path,
            output_path=args.output_path,
            compound_name=args.compound_name,
        )

        logger.info("Conversion completed successfully")

    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        raise


if __name__ == "__main__":
    main()
