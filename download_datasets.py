import argparse
import json
import shutil
import sys
import time
import urllib
import zipfile
from pathlib import Path, PurePosixPath

import gdown
import numpy as np
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_TMP_ROOT = SCRIPT_DIR / "data_tmp"
DATA_REF_ROOT = DATA_TMP_ROOT if DATA_TMP_ROOT.exists() else SCRIPT_DIR / "data"
PETA_MAPPING_PATH = SCRIPT_DIR / "peta_file_mapping.txt"


def path_has_files(path):
    return path.exists() and any(path.iterdir())


def stage_reference_metadata(dataset_path):
    readme_source = DATA_REF_ROOT / "README.md"
    if readme_source.exists():
        shutil.copy2(readme_source, dataset_path / "README.md")

    annotations_source = DATA_REF_ROOT / "annotations"
    annotations_destination = dataset_path / "annotations"
    same_dir = annotations_destination.exists() and annotations_destination.samefile(annotations_source)
    if annotations_source.exists() and not same_dir:
        if annotations_destination.exists():
            shutil.rmtree(annotations_destination)
        shutil.copytree(annotations_source, annotations_destination)

    pa100k_annotation = annotations_destination / "specialization" / "train_pa100k.json"
    specialization_dir = annotations_destination / "specialization"
    if specialization_dir.exists() and not pa100k_annotation.exists():
        legacy_pa_annotations = sorted(
            path
            for path in specialization_dir.glob("train_*.json")
            if path.name not in {"train_market.json", "train_peta.json", "train_upar_all.json", "train_pa100k.json"}
        )
        if legacy_pa_annotations:
            legacy_pa_annotations[0].rename(pa100k_annotation)

    for json_path in [pa100k_annotation, annotations_destination / "specialization" / "train_upar_all.json"]:
        if not json_path.exists():
            continue
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for image_info in payload.get("images", []):
            file_name = image_info.get("file_name")
            if not isinstance(file_name, str):
                continue
            file_path = PurePosixPath(file_name)
            if len(file_path.parts) >= 2 and file_path.parts[1] == "images" and file_path.parts[0] != "pa100k":
                image_info["file_name"] = str(PurePosixPath("pa100k", *file_path.parts[1:]))
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)


def download_url(url, dst):
    """Downloads file from a url to a destination.

    Args:
        url (str): url to download file.
        dst (str): destination path.
    """

    print(f'* url="{url}"')
    print(f'* destination="{dst}"')

    def _reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time + 1e-6
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size + 1e-6)
        sys.stdout.write(
            "\r...%d%%, %d MB, %d KB/s, %d seconds passed" % (percent, progress_size / (1024 * 1024), speed, duration)
        )
        sys.stdout.flush()

    if dst.exists():
        return
    else:
        urllib.request.urlretrieve(url, dst, _reporthook)
        sys.stdout.write("\n")


def extract_zip(src, dst):
    with zipfile.ZipFile(src, "r") as zf:
        for member in tqdm(zf.infolist(), desc="Extracting "):
            try:
                zf.extract(member, dst)
            except zipfile.error as err:
                print(err)


def prepare_market(dataset_path):
    # Market 1501 dataset
    market_path = dataset_path / "market"
    if market_path.exists():
        return
    market_1501_zipfile = dataset_path / "market_1501.zip"
    url = "https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?resourcekey=0-8nyl7K9_x37HlQm34MmrYQ"
    print("Download Market 1501 dataset")
    if not market_1501_zipfile.exists():
        gdown.download(url, output=str(market_1501_zipfile), quiet=False, use_cookies=False)
    else:
        print("Market 1501 archive already exists, skipping download")
    print("Extract Market 1501 dataset")
    extracted_market_path = dataset_path / "Market-1501-v15.09.15"
    alt_market_path = dataset_path / "Market1501"
    if not extracted_market_path.exists() and not alt_market_path.exists():
        extract_zip(market_1501_zipfile, dataset_path)
    source_market_path = extracted_market_path if extracted_market_path.exists() else alt_market_path
    if source_market_path.exists():
        if market_path.exists():
            shutil.rmtree(market_path)
        source_market_path.rename(market_path)


def prepare_pa100k(dataset_path):
    # PA-100K dataset
    pa100k_raw_path = dataset_path / "pa100k_download"
    pa100k_raw_path.mkdir(parents=True, exist_ok=True)
    pa100k_release_path = pa100k_raw_path / "release_data" / "release_data"
    pa100k_path = dataset_path / "pa100k"
    pa100k_images_path = pa100k_path / "images"
    if pa100k_images_path.exists():
        return
    print("Download PA100k dataset")
    url = "https://drive.google.com/drive/folders/1d_D0Yh7C262gr0ef9EqkvG_M3fqgAWa2?usp=sharing"
    if not (pa100k_raw_path / "annotation.zip").exists() or not (pa100k_raw_path / "data.zip").exists():
        gdown.download_folder(url, output=str(pa100k_raw_path), quiet=False, use_cookies=False)
    else:
        print("PA100k archives already exist, skipping download")
    print("Extract PA100k dataset")
    if not path_has_files(pa100k_release_path):
        extract_zip(pa100k_raw_path / "data.zip", pa100k_raw_path)
    pa100k_path.mkdir(parents=True, exist_ok=True)
    pa100k_release_path.rename(pa100k_images_path)


def prepare_peta(dataset_path):
    # PETA dataset
    peta_raw_path = dataset_path / "peta_download"
    peta_path = dataset_path / "peta"
    if peta_path.exists():
        nested_raw_path = peta_path / "PETA dataset"
        if nested_raw_path.exists():
            shutil.rmtree(nested_raw_path)
        if peta_raw_path.exists():
            shutil.rmtree(peta_raw_path)
        return
    peta_raw_path.mkdir(parents=True, exist_ok=True)
    peta_zipfile = peta_raw_path / "peta.zip"
    print("Download PETA dataset")
    url = "https://www.dropbox.com/s/52ylx522hwbdxz6/PETA.zip?dl=1"
    if not peta_zipfile.exists():
        download_url(url, peta_zipfile)
    else:
        print("PETA archive already exists, skipping download")
    print("Extract PETA dataset")
    peta_images_path = peta_raw_path / "images"
    if not peta_images_path.exists():
        extract_zip(peta_zipfile, peta_raw_path)
        mapping = {row[0]: row[1] for row in np.genfromtxt(PETA_MAPPING_PATH, dtype=str, delimiter=",")}
        for file in tqdm(peta_raw_path.glob("*/*/*/*")):
            if file.suffix == ".txt":
                continue
            source = "PETA/" + file.relative_to(peta_raw_path).as_posix()
            destination_file = peta_path / "images" / PurePosixPath(mapping[source]).name
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(file, destination_file)
    if peta_raw_path.exists():
        shutil.rmtree(peta_raw_path)


def prepare_datasets(path):
    # Datasets folder
    dataset_path = Path(path)
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Download & extract datasets
    prepare_market(dataset_path)
    prepare_pa100k(dataset_path)
    prepare_peta(dataset_path)
    stage_reference_metadata(dataset_path)

    for leftover in [
        dataset_path / "Market1501",
        dataset_path / "Market-1501-v15.09.15",
        dataset_path / "pa100k_download",
        dataset_path / "peta_download",
        dataset_path / "market_1501.zip",
    ]:
        if leftover.is_dir():
            shutil.rmtree(leftover)
        elif leftover.exists():
            leftover.unlink()

    for legacy_dir in dataset_path.glob("pa*"):
        if legacy_dir.is_dir() and legacy_dir.name != "pa100k":
            shutil.rmtree(legacy_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the public datasets used by the UPAR 2027 Challenge - Track 3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Dataset directory. Downloaded datasets are stored in this directory.",
    )
    args = parser.parse_args()

    prepare_datasets(args.data_dir)
