"""
launch_modal.py

This tool is designed to launch scripts using https://modal.com/

It sets up the necessary environment, including GPU resources and python dependencies,
and executes the specified training script remotely.

### Setup and Usage
```bash
pip install modal
modal setup  # authenticate with Modal
export HF_TOKEN="your_huggingface_token"  # if using a gated model such as llama3
modal run --detach infra/run_on_modal.py --command "torchrun --standalone --nproc_per_node=8 train.py --wandb_log=True"
```

For iterative development, consider using `modal.Volume` to cache the model between runs to
avoid redownloading the weights at the beginning of train.py
"""

import os

import modal
from modal import FilePatternMatcher

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
print(project_root)

app = modal.App("nanogpt-flash")
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .pip_install("huggingface_hub[hf_transfer]")  # enable hf_transfer for faster downloads
    .add_local_dir(project_root, "/root/", ignore=~FilePatternMatcher("**/*.py"))  # copy root of nanogpt-flash to /root/ (only python files)
)
# Define Modal volume to store training data and models
volume = modal.Volume.from_name("tinystories-data")

@app.function(
    gpu="L40S:2",
    image=image,
    volumes={"/data": volume},
    timeout=24 * 60 * 60,  # 1 day
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret")
    ],
    network_file_systems={  # add persistent storage for HF cache
        "/root/models": modal.NetworkFileSystem.from_name("hf-cache", create_if_missing=True)
    },
)
def run_command(command: str):
    """This function will be run remotely on the specified hardware."""
    import shlex
    import subprocess

    # configure HF cache directory, enable faster hf hub transfers
    os.environ["HF_HOME"] = "/root/models"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # Debug: List files in container
    print("[CONTAINER] Contents of /root:")
    subprocess.run(["ls", "-l", "/root"], check=True)

    print("[CONTAINER] Checking for train.py:")
    subprocess.run(["ls", "-l", "/root/train_gpt.py"], check=True)

    print(f"Running command: {command}")
    args = shlex.split(command)
    subprocess.run(args, check=True, cwd="/root", env=os.environ.copy())


@app.local_entrypoint()
def main(command: str):
    """Run a command remotely on modal.
    ```bash
    export HF_TOKEN="your_huggingface_token"  # if using a gated model such as llama3
    modal run --detach infra/run_on_modal.py --command "torchrun --standalone --nproc_per_node=8 train.py"
    ```
    """
    run_command.remote(command=command)
