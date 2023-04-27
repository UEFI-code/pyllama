import torch
import torch.nn as nn
import os
import json
from pathlib import Path
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from theHack.BadTransformerLLM import myBadTransfomer

class BlankModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        pass

def load(
    tokenizer_path: str,
):

    tokenizer = Tokenizer(model_path=tokenizer_path)
    model = BlankModel()
    model.params = ModelArgs()
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
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 1024,
):
    generator = load(tokenizer_path)
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
    parser.add_argument(
        "--tokenizer_path", type=str, default="llama/pyllama_data/tokenizer.model"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run(
        tokenizer_path=args.tokenizer_path,
        temperature=0.8,
        top_p=0.95,
        max_seq_len=1024,
    )
