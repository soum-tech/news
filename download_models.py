"""
Download the models the news pipeline uses.

    python news/download_models.py          the embedding model for search (BAAI/bge-small-en-v1.5, ~130 MB)
                                            -> news/models/bge-small-en-v1.5
    python news/download_models.py --llm    also the language model (Qwen/Qwen3.5-0.8B, ~1.7 GB), if it is not
                                            already in graphrag/models/Qwen3.5-0.8B
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Download the models")
    ap.add_argument("--llm", action="store_true", help="also download Qwen/Qwen3.5-0.8B")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download

    if (config.EMBED_MODEL_DIR / "config.json").exists():
        print(f"Embedding model already in {config.EMBED_MODEL_DIR}")
    else:
        print(f"Downloading {config.EMBED_MODEL_ID} -> {config.EMBED_MODEL_DIR}")
        snapshot_download(repo_id=config.EMBED_MODEL_ID, local_dir=str(config.EMBED_MODEL_DIR),
                          allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*", "modules.json"])
    if a.llm:
        target = config.REPO / "models" / "Qwen3.5-0.8B"
        if (target / "config.json").exists():
            print(f"Language model already in {target}")
        else:
            print(f"Downloading Qwen/Qwen3.5-0.8B -> {target}")
            snapshot_download(repo_id="Qwen/Qwen3.5-0.8B", local_dir=str(target))
    print("Done")


if __name__ == "__main__":
    main()
