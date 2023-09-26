from pathlib import Path

from lit_model import LitModel

from utils.inference import inference_model


def main(
    checkpoint: Path = Path(
        "checkpoints/trained/bistro/09-26+16:56:58/epoch=0-step=21636-val_loss=0.55.ckpt"
    ),
    temperature: float = 0.7,
    interactive: bool = False,
):
    print(checkpoint, temperature, interactive)

    model = LitModel.load_from_checkpoint(checkpoint)
    tokenizer = model.tokenizer
    model.eval()

    if not interactive:
        inference_model(model, idx=[0], temperature=temperature, max_new_tokens=100)

    else:
        # Interactive mode
        while True:
            print("\n")
            start = input("Enter text to run through the model: ")
            if start.lower() == "quit" or start.lower() == "exit":
                break
            print("\n")
        print("TODO")  # TODO


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
