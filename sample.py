import torch

from pathlib import Path

from lit_model import LitModel


from utils.inference import inference_model

from model import GPT

device = "cuda"


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

    new_state_dict = {
        key.replace("model.", ""): value for key, value in ckpt["state_dict"].items()
    }

    print(ckpt["state_dict"].keys(), new_state_dict.keys())

    # TODO: should strict=True
    # TODO: hmm state dict has .model in front of everything
    model.load_state_dict(new_state_dict, strict=False, assign=True)

    model.eval()
    model.to(device)

    if not interactive:
        inference_model(
            model,
            idx=tokenizer.encode(
                "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ 737 * 850 = ASSISTANT:",
                device=torch.device("meta"),
            ),
            temperature=temperature,
            max_new_tokens=69,
        )

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
