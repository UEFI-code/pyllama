import torch
import os
import json
from pathlib import Path
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from theHack.BadTransformerLLM import myBadTransfomer

def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    if os.path.exists('theHack/BadTransformer'):
        os.chdir('theHack')
    else:
        print("Please run theHack_*.py first")
        exit(0)
    
    model.tok_embeddings = torch.load('theTokEmbeddingLayer.pth', map_location='cpu')
    print('Loaded theTokEmbeddingLayer.pth')
    model.norm = torch.load('theNormLayer.pth', map_location='cpu')
    print('Loaded theNormLayer.pth')
    model.output = torch.load('theOutputLayer.pth', map_location='cpu')
    print('Loaded theOutputLayer.pth')
    os.chdir('BadTransformer')
    model.badlayers = []
    for i in range(32):
        model.badlayers.append(torch.load('theBadTransformerLayer%d.pth' % i, map_location='cpu'))
        print('Loaded theBadTransformerLayer%d.pth' % i)

    def theForward(self, x):
        print('Hooked the forward function')
        x = self.tok_embeddings(x)
        for layer in self.badlayers:
            x = layer(x)
        x = self.norm(x)
        x = self.output(x[:, -1, :])
        return x
    
    model.forward = theForward.__get__(model, Transformer)

    generator = LLaMA(model, tokenizer, useGPU=False)
    return generator


def run(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 1024,
    max_batch_size: int = 1,
):
    local_rank = 0
    world_size = 1
    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )
    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",  # removed: keep only one prompt
    ]
    while True:
        print("Prompt:", prompts)
        results = generator.generate(
            prompts, max_gen_len=256, temperature=temperature, top_p=top_p
        )
        for result in results:
            print("🦙LLaMA:", result.strip())

        user_input = input("please enter your prompts (Ctrl+C to exit): ")
        prompts = [user_input]


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="llama/pyllama_data/7B")
    parser.add_argument(
        "--tokenizer_path", type=str, default="llama/pyllama_data/tokenizer.model"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        temperature=0.8,
        top_p=0.95,
        max_seq_len=1024,
        max_batch_size=1,
    )
