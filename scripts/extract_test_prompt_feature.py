import torch
import os
import numpy as np
import libs.autoencoder
import libs.clip
from datasets import MSCOCODatabase
import argparse
from tqdm import tqdm


def main():
    prompts = [
        'A green train is coming down the tracks.',
        'A group of skiers are preparing to ski down a mountain.',
        'A small kitchen with a low ceiling.',
        'A group of elephants walking in muddy water.',
        'A living area with a television and a table.',
        'A road with traffic lights, street lights and cars.',
        'A bus driving in a city area with traffic signs.',
        'A bus pulls over to the curb close to an intersection.',
        'A group of people are walking and one is holding an umbrella.',
        'A baseball player taking a swing at an incoming ball.',
        'A city street line with brick buildings and trees.',
        'A close up of a plate of broccoli and sauce.',
    ]

    device = 'cuda'
    clip = libs.clip.FrozenCLIPEmbedder()
    clip.eval()
    clip.to(device)

    save_dir = f'assets/datasets/coco256_features/run_vis'
    latent = clip.encode(prompts)
    for i in range(len(latent)):
        c = latent[i].detach().cpu().numpy()
        np.save(os.path.join(save_dir, f'{i}.npy'), (prompts[i], c))


if __name__ == '__main__':
    main()
