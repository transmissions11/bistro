from pathlib import Path


# TODO: why doesnt star work
def main(checkpoint: Path, temperature: float = 0.7, interactive: bool = False):
    print(checkpoint, temperature, interactive)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
