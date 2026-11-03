import argparse
import asyncio
from pathlib import Path

from ratatosk.app import run


def main() -> None:
    parser = argparse.ArgumentParser(prog="ratatosk", description="GPU traffic control for Idun")
    parser.add_argument("--config", default="data/settings.json", help="path to the settings file")
    args = parser.parse_args()
    try:
        asyncio.run(run(Path(args.config)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
