"""
TinyStories Dataset
https://huggingface.co/datasets/roneneldan/TinyStories
Downloads and tokenizes the data and saves the data shards to disk
"""

import os
import modal
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Define Modal App
app = modal.App("nanogpt_mla")
# Define image for our container
image = modal.Image.debian_slim().pip_install(
    "datasets",
    "numpy",
    "tiktoken"
)
# Define Modal volume to store training data and models
volume = modal.Volume.from_name("tinystories-data", create_if_missing=True)

def tokenize(doc):
    # Init the Tokenizer
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"] # End of text token

    # Tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

@app.function(
    gpu="T4",
    image=image,
    volumes={"/data": volume},
    timeout=86400
)
def data_prep(
    data_dir="tinystories",
    shard_size=int(5e7),
):
    # Create the cache directory
    cache_dir = os.path.join("/data", data_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # Download the dataset
    wikitext = load_dataset("roneneldan/TinyStories", split="train")

    # Tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # Pre-allocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize, wikitext, chunksize=16):
            # is there enough space in the current shard for the new tokens
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))

            else:
                # write the current shard and start a new one
                split = "valid" if shard_index == 0 else "train"
                filename = os.path.join(cache_dir, f"tinystories_{split}_{shard_index:06d}")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                np.save(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # Write any reamining tokens as the last shard
        if token_count != 0:
            split = "valid" if shard_index == 0 else "train"
            filename = os.path.join(cache_dir, f"tinystories_{split}_{shard_index:06d}")
            np.save(filename, all_tokens_np[:token_count])

        print("\nFile saved in volume:")
        for f in os.listdir(cache_dir):
            print(f"- {f}")

@app.local_entrypoint()
def main():
    data_prep.remote()

if __name__=="__main__":
    main()
