# bistro

Opinionated GPT implementation and finetuning harness.

## Installation

Cloning the repo:

```sh
git clone https://github.com/transmisisons11/bistro
cd bistro
```

Installing dependencies:

```sh
pip install -r requirements.txt
```

Downloading and setting up a base model:

```sh
python lit_script.py download --repo_id lmsys/vicuna-7b-v1.5
python lit_script.py download convert_hf_checkpoint --checkpoint_dir checkpoints/lmsys/vicuna-7b-v1.5 --dtype bfloat16
```

_(Optional)_ Use Flash Attention 2 (only available in PyTorch 2.2)

```bash
pip uninstall -y torch
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

See [lit-gpt's setup guide](https://github.com/Lightning-AI/lit-gpt#setup) for additional info.

## Acknowledgements

Built on [lit-gpt](https://github.com/Lightning-AI/lit-gpt).
