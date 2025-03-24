# fetches pretraining data, tools for validating that the data has been unpacked.
from datasets import config, load_dataset
import os
BOLD = '\033[1m'

hf_hub_tarball = ["abc-vg-instruct.tar.gz"]

def download_finetuning_data():
    ds = load_dataset("TIGER-Lab/ABC-VG-Instruct")
    cache_dir = config.HF_DATASETS_CACHE
    print(f"{BOLD}Downloading instruction finetuning data, this could take a while...{BOLD}")
    path = os.path.join(cache_dir, "tigerlab")
    os.makedirs(path, exist_ok=True)
    from huggingface_hub import hf_hub_download
    for tarball in hf_hub_tarball:
        hf_hub_download(repo_id="TIGER-Lab/ABC-VG-Instruct", repo_type="dataset", filename=tarball, local_dir=path)

    print(f"{BOLD}Unpacking instruction finetuning data, this could take a while...{BOLD}")
    os.system(f"cat {path}/abc-vg-instruct.tar.gz | tar -xvzf - -C {path} && rm {path}/abc-vg-instruct.tar.gz")


def check_finetuning_downloaded():
    img_path = get_finetuning_location()
    return os.path.isdir(img_path)

def get_finetuning_location():
    from datasets import config
    import os
    cache_dir = config.HF_DATASETS_CACHE
    return os.path.join(cache_dir, "tigerlab/ABC-VG-Instruct")


if __name__ == "__main__":
    download_finetuning_data()
