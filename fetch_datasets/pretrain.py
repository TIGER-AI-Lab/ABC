# fetches
from datasets import config
import os
cache_dir = config.HF_DATASETS_CACHE
BOLD = '\033[1m'

hf_hub_tarball = ["abc-pretrain.tar.gz.part_aa",
                "abc-pretrain.tar.gz.part_ab",
                "abc-pretrain.tar.gz.part_ac",
                "abc-pretrain.tar.gz.part_ad",
                "abc-pretrain.tar.gz.part_ae",
                "abc-pretrain.tar.gz.part_af",
                "abc-pretrain.tar.gz.part_ag",
                "abc-pretrain.tar.gz.part_ah",
                "abc-pretrain.tar.gz.part_ai"]

def download_pretraining_data():
    print(f"{BOLD}Downloading pretraining data, this could take a while...{BOLD}")
    path = os.path.join(cache_dir, "tigerlab/abc-pretrain")
    os.makedirs(path, exist_ok=True)
    from huggingface_hub import hf_hub_download
#    for tarball in hf_hub_tarball:
#        hf_hub_download(repo_id="TIGER-Lab/ABC-Pretraining-Data", repo_type="dataset", filename=tarball, local_dir=path)

    print(f"{BOLD}Unpacking pretraining data, this could take a while...{BOLD}")
    os.system(f"cat {path}/abc-pretrain.tar.gz.part_* | tar -xvzf - -C {path} && rm {path}/abc-pretrain.tar.gz.part_*")


def check_pretraining_downloaded():
    from datasets import config
    import os
    cache_dir = config.HF_DATASETS_CACHE
    img_path = os.path.join(cache_dir, "tigerlab/abc-pretrain/train")
    return os.path.isdir(img_path)

if __name__ == "__main__":
    download_pretraining_data()
