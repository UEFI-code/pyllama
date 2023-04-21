import torch
from pathlib import Path
import json
from llama import Transformer, ModelArgs, Tokenizer
import os

def load(
    ckpt_dir = 'llama/pyllama_data/7B',
    tokenizer_path = 'llama/pyllama_data/tokenizer.model',
    local_rank = 0,
    world_size = 1,
    max_seq_len = 1024,
    max_batch_size = 1,
):
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    #torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)
    return model

theModel = load()

if not os.path.exists("theHack"):
    os.mkdir("theHack")
os.chdir("theHack")
