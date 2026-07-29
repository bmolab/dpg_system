"""
Download vision models to permanent local directories.

Usage
    python dpg_system/download_vision_models.py                  # download all
    python dpg_system/download_vision_models.py moondream        # just Moondream2
    python dpg_system/download_vision_models.py smol             # just SmolVLM 500M
    python dpg_system/download_vision_models.py smol 256M        # SmolVLM 256M
    python dpg_system/download_vision_models.py qwen             # just Qwen2.5-VL-3B
    python dpg_system/download_vision_models.py gemma            # just Gemma 4 E4B
    python dpg_system/download_vision_models.py gemma E2B        # Gemma 4 E2B

Models are saved to dpg_system/models/<model_name>/
"""

import os
import sys


MODELS = {
    'moondream': {
        'id': 'vikhyatk/moondream2',
        'dir': 'moondream2',
        'desc': 'Moondream2 (~3.8 GB)',
        'type': 'moondream',
    },
    'smol': {
        '256M': {
            'id': 'HuggingFaceTB/SmolVLM-256M-Instruct',
            'dir': 'smolvlm-256m',
            'desc': 'SmolVLM 256M (~0.5 GB)',
            'type': 'vision2seq',
        },
        '500M': {
            'id': 'HuggingFaceTB/SmolVLM-500M-Instruct',
            'dir': 'smolvlm-500m',
            'desc': 'SmolVLM 500M (~1 GB)',
            'type': 'vision2seq',
        },
        '2.2B': {
            'id': 'HuggingFaceTB/SmolVLM-Instruct',
            'dir': 'smolvlm-2.2b',
            'desc': 'SmolVLM 2.2B (~4.4 GB)',
            'type': 'vision2seq',
        },
    },
    'qwen': {
        'id': 'Qwen/Qwen2.5-VL-3B-Instruct',
        'dir': 'qwen25-vl-3b',
        'desc': 'Qwen2.5-VL-3B (~6 GB)',
        'type': 'qwen',
    },
    'gemma': {
        'E2B': {
            'id': 'google/gemma-4-E2B-it',
            'dir': 'gemma4-e2b',
            'desc': 'Gemma 4 E2B (~4 GB)',
            'type': 'gemma',
        },
        'E4B': {
            'id': 'google/gemma-4-E4B-it',
            'dir': 'gemma4-e4b',
            'desc': 'Gemma 4 E4B (~8 GB)',
            'type': 'gemma',
        },
    },
}


def download_model(model_id, save_dir, desc, model_type):
    if os.path.exists(save_dir) and os.listdir(save_dir):
        print(f'  {desc} already exists at {save_dir}')
        resp = input('  Re-download? [y/N] ').strip().lower()
        if resp != 'y':
            return

    print(f'  Downloading {desc} …')
    os.makedirs(save_dir, exist_ok=True)

    # Download all files with visible progress bars
    from huggingface_hub import snapshot_download
    print(f'  Fetching {model_id} …')
    snapshot_dir = snapshot_download(model_id)
    print(f'  Snapshot cached at: {snapshot_dir}')

    # Copy all files from the snapshot to our permanent directory
    import shutil
    print(f'  Copying files to {save_dir} …')
    count = 0
    for item in os.listdir(snapshot_dir):
        src = os.path.join(snapshot_dir, item)
        dst = os.path.join(save_dir, item)
        if item.startswith('.'):
            continue  # skip .gitattributes etc
        if os.path.isfile(src):
            # Resolve symlinks (HF cache uses symlinks to blobs)
            real_src = os.path.realpath(src)
            shutil.copy2(real_src, dst)
            size_mb = os.path.getsize(dst) / 1e6
            print(f'    {item} ({size_mb:.1f} MB)')
            count += 1
        elif os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f'    {item}/')
            count += 1

    print(f'  Copied {count} files to {save_dir}\n')


def main():
    base_dir = os.path.join(os.path.dirname(__file__), 'models')
    args = sys.argv[1:]

    if not args:
        targets = ['moondream', 'smol', 'qwen', 'gemma']
    else:
        targets = [args[0]]

    for target in targets:
        if target == 'moondream':
            info = MODELS['moondream']
            save_dir = os.path.join(base_dir, info['dir'])
            download_model(info['id'], save_dir, info['desc'], info['type'])

        elif target == 'smol':
            size = args[1] if len(args) > 1 else '500M'
            if size not in MODELS['smol']:
                print(f'Unknown SmolVLM size: {size}. Choose from: 256M, 500M, 2.2B')
                continue
            info = MODELS['smol'][size]
            save_dir = os.path.join(base_dir, info['dir'])
            download_model(info['id'], save_dir, info['desc'], info['type'])

        elif target == 'qwen':
            info = MODELS['qwen']
            save_dir = os.path.join(base_dir, info['dir'])
            download_model(info['id'], save_dir, info['desc'], info['type'])

        elif target == 'gemma':
            size = args[1] if len(args) > 1 else 'E4B'
            if size not in MODELS['gemma']:
                print(f'Unknown Gemma 4 size: {size}. Choose from: E2B, E4B')
                continue
            info = MODELS['gemma'][size]
            save_dir = os.path.join(base_dir, info['dir'])
            download_model(info['id'], save_dir, info['desc'], info['type'])

        else:
            print(f'Unknown model: {target}. Choose from: moondream, smol, qwen, gemma')

    print('Done!')


if __name__ == '__main__':
    main()
