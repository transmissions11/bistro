import torch

from pathlib import Path

from lit_model import LitModel


from utils.inference import inference_model

from model import GPT


def main(*, checkpoint: Path, temperature: float = 0.7, interactive: bool = False):
    print(checkpoint, temperature, interactive)

    ckpt = torch.load(str(checkpoint), mmap=True)

    hparams = ckpt["hyper_parameters"]
    tokenizer = hparams["tokenizer"]

    # todo seed everything

    with torch.device("meta"):
        model = GPT(
            config=hparams["model_config"],
            soft_prompt_tkn=tokenizer.token_to_id(hparams["soft_prompt_tkn"]),
            num_soft_prompt_tkns=hparams["num_soft_prompt_tkns"],
        )

    # TODO: should strict=True
    model.load_state_dict(ckpt["state_dict"], strict=False, assign=True)

    # checkpoint = torch.load(checkpoint)

    # model = LitModel.load_from_checkpoint(checkpoint, strict=False)
    # tokenizer = model.tokenizer
    # model.eval()

    # if not interactive:
    #     inference_model(model, idx=[0], temperature=temperature, max_new_tokens=100)

    # else:
    #     # Interactive mode
    #     while True:
    #         print("\n")
    #         start = input("Enter text to run through the model: ")
    #         if start.lower() == "quit" or start.lower() == "exit":
    #             break
    #         print("\n")
    #     print("TODO")  # TODO


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
