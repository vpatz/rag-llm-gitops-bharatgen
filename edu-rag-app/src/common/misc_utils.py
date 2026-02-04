import hashlib
import logging
import os
import json
from pathlib import Path

LOG_LEVEL = logging.INFO
SETTINGS_PATH = Path(__file__).resolve().parent.parent / "settings.json"
with open(SETTINGS_PATH, 'r') as f:
    data = json.load(f)
LOCAL_CACHE_DIR = data.get("local_cache_dir", "")
chunk_suffix = "_clean_chunk.json"
text_suffix = "_clean_text.json"
table_suffix = "_tables.json"
image_suffix = "_images.json"

def set_log_level(level):
    global LOG_LEVEL
    LOG_LEVEL = level

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)-18s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


def get_txt_tab_img_filenames(file_paths, out_path):
    original_filenames = [fp.split('/')[-1] for fp in file_paths]
    input_txt_files, input_tab_files, input_img_files = [], [], []
    for fn in original_filenames:
        f, _ = os.path.splitext(fn)
        input_txt_files.append(f'{out_path}/{f}{text_suffix}')
        input_tab_files.append(f'{out_path}/{f}{table_suffix}')
        input_img_files.append(f'{out_path}/{f}{image_suffix}')
    return original_filenames, input_txt_files, input_tab_files, input_img_files


def get_model_endpoints(json_file_path=SETTINGS_PATH):
    # Open and read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract values for each model from the JSON data
    emb_model_dict = {
        'emb_endpoint': data.get("emb_endpoint", ""),
        'emb_model':    data.get("emb_model", ""),
        'max_tokens':   int(data.get("emb_max_tokens", 512)),  # Default to 512 if not found
    }

    llm_model_dict = {
        'llm_endpoint': data.get("llm_endpoint", ""),
        'llm_model':    data.get("llm_model", ""),
    }

    vlm_model_dict = {
        'vlm_endpoint': data.get("vlm_endpoint", ""),
        'vlm_model':    data.get("vlm_model", ""),
    }

    reranker_model_dict = {
        'reranker_endpoint': data.get("reranker_endpoint", ""),
        'reranker_model':    data.get("reranker_model", ""),
    }

    return emb_model_dict, llm_model_dict, vlm_model_dict, reranker_model_dict

def setup_docs_dir(dir_name):
    raw_dir = os.path.join(LOCAL_CACHE_DIR, dir_name)
    os.makedirs(raw_dir, exist_ok=True)
    return raw_dir

def setup_cache_dir(dir):
    cache_dir = os.path.join(LOCAL_CACHE_DIR, f'{dir}_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def generate_file_checksum(file):
    sha256 = hashlib.sha256()
    with open(file, 'rb') as f:
        for chunk in iter(lambda: f.read(128 * sha256.block_size), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def verify_checksum(file, checksum_file):
    file_sha256 = generate_file_checksum(file)
    f = open(checksum_file, "r")
    data = f.read()
    csum = data.split(' ')[0]
    if csum == file_sha256:
        return True
    return False

def get_unprocessed_files(original_files, processed_chunk_files):
    processed_pdfs = []
    for file in processed_chunk_files:
        path = Path(file)
        file = path.name
        processed_pdfs.append(file.replace(chunk_suffix, ".pdf"))

    original_file_names = []
    for file in original_files:
        path = Path(file)
        file = path.name
        original_file_names.append(file)

    return set(original_file_names).difference(set(processed_pdfs))
