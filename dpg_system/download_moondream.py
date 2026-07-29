"""
Download and save Moondream2 model to a permanent local directory.

Run this once while connected to the internet:
    python dpg_system/download_moondream.py

The model will be saved to dpg_system/models/moondream2/
"""

import os
import sys

def main():
    model_id = 'vikhyatk/moondream2'
    save_dir = os.path.join(os.path.dirname(__file__), 'models', 'moondream2')

    if os.path.exists(save_dir) and os.listdir(save_dir):
        print(f'Model already exists at {save_dir}')
        resp = input('Re-download? [y/N] ').strip().lower()
        if resp != 'y':
            print('Skipping download.')
            return

    print(f'Downloading {model_id} …')
    print('This may take a few minutes (~4 GB).\n')

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f'\nModel saved to {save_dir}')
    print('You can now use vision_describe offline.')


if __name__ == '__main__':
    main()
