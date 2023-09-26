from pathlib import Path


# TODO: why doesnt star work
def main(checkpoint: Path, temperature: float, interactive: bool):
    print(checkpoint, temperature, interactive)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
