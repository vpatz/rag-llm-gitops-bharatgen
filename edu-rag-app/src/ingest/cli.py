import logging
import os
import time
import json
from pathlib import Path
import argparse
import shutil

from ..common.misc_utils import *
from ..common.db_utils import MilvusVectorStore

language_dict = {"English": "en", "Hindi": "hi"}


def reset_db():
    vector_store = MilvusVectorStore()
    vector_store.reset_collection()
    logger.info("✅ DB Cleaned successfully!")


def extract_metadata_from_chunks(chunk):
    """Metadata already exists inside chunk"""
    return {
        "language": chunk.get("language", ""),
        "subject": chunk.get("subject", ""),
        "class_level": chunk.get("class_level", "")
    }

def ignore_json_files(src, names):
    return [name for name in names if name == 'chunks.json']

def copy_sub_dir(source_dir: Path, destination_dir: Path):
    destination_dir.mkdir(parents=True, exist_ok=True)

    for item in source_dir.iterdir():
        if item.is_dir():
            dest_subdir = destination_dir / item.name

            shutil.copytree(
                item,
                dest_subdir,
                ignore=ignore_json_files,
                dirs_exist_ok=True
            )


def ingest(directory_path):

    logger.info(f"Ingestion started from dir '{directory_path}'")

    # 🔍 Find all chunks.json recursively
    chunk_files = list(Path(directory_path).rglob("chunks.json"))

    if not chunk_files:
        logger.info(f"No chunks.json files found in '{directory_path}'")
        return

    logger.info(f"Found {len(chunk_files)} chunk files")

    emb_model_dict, _, _, _ = get_model_endpoints()

    vector_store = MilvusVectorStore()
    collection_name = vector_store._generate_collection_name()
    cache_path = setup_cache_dir(collection_name)
    copy_sub_dir(Path(directory_path), Path(cache_path))

    start_time = time.time()
    total_chunks = 0
    total_images = 0
    total_tables = 0

    per_file_stats = {}

    for chunk_file in chunk_files:
        logger.info(f"Loading chunks from {chunk_file}")

        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not chunks:
            continue

        # Count stats
        file_name = chunks[0].get("filename", Path(chunk_file).parent.name)

        num_chunks = len(chunks)
        num_images = sum(1 for c in chunks if c.get("type") == "image")
        num_tables = sum(1 for c in chunks if c.get("type") == "table")

        total_chunks += num_chunks
        total_images += num_images
        total_tables += num_tables

        per_file_stats[file_name] = {
            "chunk_count": num_chunks,
            "image_count": num_images,
            "table_count": num_tables
        }

        # Insert into Milvus
        vector_store.insert_chunks(
            emb_model=emb_model_dict['emb_model'],
            emb_endpoint=emb_model_dict['emb_endpoint'],
            max_tokens=emb_model_dict['max_tokens'],
            chunks=chunks
        )

    end_time = time.time()

    logger.info("✅ All chunks loaded into DB successfully!")
    logger.info(f"Total Chunks: {total_chunks}")
    logger.info(f"Total Image Chunks: {total_images}")
    logger.info(f"Total Table Chunks: {total_tables}")
    logger.info(f"⏱ Time taken: {end_time - start_time:.2f} seconds")

    if per_file_stats:
        logger.info("📊 Stats of ingested chunk files:")

        max_file_len = max(len(name) for name in per_file_stats.keys())

        header = f"| {'PDF':<{max_file_len}} | {'Chunks':^10} | {'Images':^10} | {'Tables':^10} |"
        line = "-" * len(header)

        print(line)
        print(header)
        print(line)

        for name, stats in per_file_stats.items():
            print(f"| {name:<{max_file_len}} | {stats['chunk_count']:^10} | {stats['image_count']:^10} | {stats['table_count']:^10} |")

        print(line)
        print(f"| {'Total':<{max_file_len}} | {total_chunks:^10} | {total_images:^10} | {total_tables:^10} |")
        print(line)


# ================= CLI =================

common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument("--debug", action="store_true", help="Enable debug logging")

parser = argparse.ArgumentParser(
    description="Chunk Ingestion CLI",
    formatter_class=argparse.RawTextHelpFormatter,
    parents=[common_parser]
)

command_parser = parser.add_subparsers(dest="command", required=True)

ingest_parser = command_parser.add_parser(
    "ingest",
    help="Ingest prebuilt chunks.json files into Milvus",
    parents=[common_parser]
)
ingest_parser.add_argument(
    "--path",
    type=str,
    default="/var/docs",
    help="Root directory containing chunks.json files"
)

command_parser.add_parser(
    "clean-db",
    help="Clean the Milvus DB",
    parents=[common_parser]
)


def main(argv=None):
    command_args = parser.parse_args(argv)

    log_level = logging.INFO
    env_log_level = os.getenv("LOG_LEVEL", "")
    if "debug" in env_log_level.lower() or command_args.debug:
        log_level = logging.DEBUG
    set_log_level(log_level)

    global logger
    logger = get_logger("Ingest")

    if command_args.command == "ingest":
        ingest(command_args.path)
    elif command_args.command == "clean-db":
        reset_db()
    else:
        logger.error("Unknown command: %s", getattr(command_args, "command", None))


if __name__ == "__main__":
    main()
