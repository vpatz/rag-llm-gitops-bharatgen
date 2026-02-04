from glob import glob
import os
import shutil
import numpy as np
import hashlib
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict
from pymilvus import (
    connections, utility, Collection, CollectionSchema,
    FieldSchema, DataType
)
from rank_bm25 import BM25Okapi
import pickle

import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from indicnlp import common
RESOURCES = Path(__file__).resolve().parent.parent.parent / "indic_nlp_resources"
common.set_resources_path(str(RESOURCES))
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

from ..common.emb_utils import Embedding
from ..common.misc_utils import LOCAL_CACHE_DIR, get_logger

logger = get_logger("Milvus")

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "settings.json"
if not SETTINGS_PATH.exists():
    raise FileNotFoundError(f"settings.json not found at {SETTINGS_PATH}")

with open(SETTINGS_PATH, "r") as f:
    data = json.load(f)

# Allow environment variables to override settings if set
_env_host = os.getenv("MILVUS_HOST")
_env_port = os.getenv("MILVUS_PORT")
_env_db_prefix = os.getenv("MILVUS_DB_PREFIX")
_env_cname = os.getenv("MILVUS_COLLECTION_NAME")

# Read and coerce types safely
_raw_host = _env_host if _env_host is not None else data.get("milvus_host")
raw_port = _env_port if _env_port is not None else data.get("milvus_port")
db_prefix_raw = _env_db_prefix if _env_db_prefix is not None else data.get("milvus_db_prefix")
c_name_raw = _env_cname if _env_cname is not None else data.get("milvus_collection_name")

# Ensure host is a string (or None)
HOST = None
if _raw_host is not None:
    HOST = str(_raw_host).strip()

# Ensure port is an int if possible, otherwise None
PORT = None
if raw_port is not None and str(raw_port).strip() != "":
    try:
        PORT = int(raw_port)
    except Exception:
        # allow string ports like "19530"; pymilvus accepts str too
        PORT = str(raw_port).strip()

# Coerce other values to strings (safe defaults)
DB_PREFIX = str(db_prefix_raw) if db_prefix_raw is not None else ""
C_NAME = str(c_name_raw) if c_name_raw is not None else ""


# Helper functions
en_stopwords = set(stopwords.words("english"))
en_stemmer = PorterStemmer()

def process_english(text):
    tokens = word_tokenize(text.lower())
    return [en_stemmer.stem(t) for t in tokens if t.isalnum() and t not in en_stopwords]

def load_indic_stopwords(path):
    with open(path, encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

hi_stopwords = load_indic_stopwords(Path(__file__).resolve().parent.parent / "stopword-hi.txt")

factory = IndicNormalizerFactory()
normalizer_hi = factory.get_normalizer("hi")

def process_hindi(text):
    text = normalizer_hi.normalize(text)
    tokens = indic_tokenize.trivial_tokenize(text)
    return [t for t in tokens if t.strip() and t not in hi_stopwords]

def tokenize_for_bm25(text, language):
    if language == "English":
        return process_english(text)
    elif language == "Hindi":
        return process_hindi(text)
    else:
        return text.lower().split()

def norm(x):
    return x.strip().lower() if isinstance(x, str) else x

def metadata_matches_filters(metadata, language=None, subject=None, class_level=None):
    if language and norm(metadata.get("language")) != norm(language):
        return False
    if subject and norm(metadata.get("subject")) != norm(subject):
        return False
    if class_level and norm(metadata.get("class_level")) != norm(class_level):
        return False
    return True

def build_filter_expr(language=None, subject=None, class_level=None):
        filters = []

        if language:
            filters.append(f'language == "{language}"')

        if subject:
            filters.append(f'subject == "{subject}"')

        if class_level:
            filters.append(f'class_level == "{class_level}"')

        return " and ".join(filters) if filters else None

class MilvusNotReadyError(Exception):
    pass

class MilvusVectorStore:
    def __init__(
        self,
        host=HOST,
        port=PORT,
        db_prefix=DB_PREFIX,
        c_name=C_NAME
    ):
        self.host = host
        self.port = port
        self.db_prefix = db_prefix
        self.c_name = c_name
        self.collection = None
        self.collection_name = None
        self._embedder = None
        self._embedder_config = {}
        self.page_content_corpus = []
        self.metadata_map = []
        self.bm25 = None
        self.tokenized_corpus = []

        connections.connect("default", host=self.host, port=self.port)

    def _generate_collection_name(self):
        hash_part = hashlib.md5(self.c_name.encode()).hexdigest()
        return f"{self.db_prefix}_{hash_part}"

    def _get_index_paths(self):
        base_path = os.path.join(LOCAL_CACHE_DIR, f"{self.collection_name}_bm25_index")
        return (
            f"{base_path}.pkl",
            f"{base_path}_metadata.pkl",
            f"{base_path}_corpus.pkl"
        )

    def _save_sparse_index(self):
        index_path, metadata_path, corpus_path = self._get_index_paths()

        with open(index_path, "wb") as f:
            pickle.dump(self.bm25, f)

        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata_map, f)

        with open(corpus_path, "wb") as f:
            pickle.dump(self.page_content_corpus, f)

    def _load_sparse_index(self):
        index_path, metadata_path, corpus_path = self._get_index_paths()

        if os.path.exists(index_path) and os.path.exists(metadata_path) and os.path.exists(corpus_path):
            with open(index_path, "rb") as f:
                self.bm25 = pickle.load(f)

            with open(metadata_path, "rb") as f:
                self.metadata_map = pickle.load(f)

            with open(corpus_path, "rb") as f:
                self.page_content_corpus = pickle.load(f)

            self.tokenized_corpus = [doc.split() for doc in self.page_content_corpus]

            logger.info(f"✅ Loaded BM25 index for collection '{self.collection_name}'.")
            return True

        return False

    def _setup_collection(self, name, dim):
        if utility.has_collection(name):
            return Collection(name=name)

        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="page_content", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="class_level", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="page_numbers", dtype=DataType.VARCHAR, max_length=128)
        ]

        schema = CollectionSchema(fields=fields, description="RAG chunk storage (dense only)")
        collection = Collection(name=name, schema=schema)

        collection.create_index(
            field_name="embedding",
            index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        )

        return collection

    def _ensure_embedder(self, emb_model, emb_endpoint, max_tokens):
        config = {"model": emb_model, "endpoint": emb_endpoint, "max_tokens": max_tokens}
        if self._embedder is None or self._embedder_config != config:
            logger.debug(f"⚙️ Initializing embedder: {emb_model}")
            self._embedder = Embedding(emb_model, emb_endpoint, max_tokens)
            self._embedder_config = config

    def reset_collection(self):
        name = self._generate_collection_name()
        if utility.has_collection(name):
            utility.drop_collection(name)
            logger.info(f"Collection {name} deleted.")
        else:
            logger.info(f"Collection {name} does not exist!")

        files_to_remove = glob(os.path.join(LOCAL_CACHE_DIR, name+"*"))
        if files_to_remove:
            for file_path in files_to_remove:
                try:
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        continue
                    os.remove(file_path)
                except OSError as e:
                    logger.error(f"Error removing {file_path}: {e}")
            logger.info("Local cache cleaned up.")
        else:
            logger.info("Local cache cleaned up already!")

        self.page_content_corpus = []
        self.metadata_map = []
        self.vectorizer = None
        self.sparse_matrix = None

    def insert_chunks(self, emb_model, emb_endpoint, max_tokens, chunks, batch_size=10):
        if not chunks:
            logger.debug("Nothing to chunk!")
            return

        self._ensure_embedder(emb_model, emb_endpoint, max_tokens)
        self.collection_name = self._generate_collection_name()

        sample_embedding = self._embedder.embed_documents([chunks[0]["page_content"]])[0]
        dim = len(sample_embedding)

        self.collection = self._setup_collection(self.collection_name, dim)
        self.collection.load()

        logger.debug(f"Inserting {len(chunks)} chunks into Milvus...")

        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i:i + batch_size]
            page_contents = [doc.get("page_content") for doc in batch]
            embeddings = self._embedder.embed_documents(page_contents)

            filenames = [doc.get("filename", "") for doc in batch]
            types = [doc.get("type", "") for doc in batch]
            sources = [doc.get("source", "") for doc in batch]
            languages = [doc.get("language", "") for doc in batch]
            subjects = [doc.get("subject", "") for doc in batch]
            class_levels = [doc.get("class_level", "") for doc in batch]
            chunk_ids = [np.int64(int(hashlib.md5(doc["chunk_id"].encode()).hexdigest()[:16], 16) % (2**63)) for doc in batch]
            page_numbers = [json.dumps(doc.get("page_numbers", [])) for doc in batch]

            self.collection.upsert([
                chunk_ids,
                embeddings,
                page_contents,
                filenames,
                types,
                sources,
                languages,
                subjects,
                class_levels,
                page_numbers
            ])

            self.metadata_map.extend([
                {
                    "chunk_id": cid,
                    "filename": fn,
                    "type": t,
                    "source": s,
                    "page_content": pc,
                    "language": l,
                    "subject": sub,
                    "class_level": cl,
                    "page_numbers": pn
                }
                for cid, fn, t, s, pc, l, sub, cl, pn in zip(
                    chunk_ids, filenames, types, sources, page_contents, languages, subjects, class_levels, page_numbers
                )
            ])

        logger.debug("Building BM25 index")
        if not self.page_content_corpus:
            self._load_sparse_index()  # load existing corpus if any

        self.page_content_corpus.extend(page_contents)

        self.tokenized_corpus = [
            tokenize_for_bm25(doc, meta.get("language", "English"))
            for doc, meta in zip(self.page_content_corpus, self.metadata_map)
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._save_sparse_index()
        logger.debug(f"Inserted the chunks into collection.")

    def _rrf_fusion(self, dense_results, sparse_results, top_k):
        """
        Perform Reciprocal Rank Fusion (RRF) on dense and sparse results.
        Each result should be a list of dicts with at least 'chunk_id' field.
        """
        rrf_k = 60  # RRF constant to dampen higher ranks
        score_map = defaultdict(float)
        doc_map = {}

        # Process dense results
        for rank, doc in enumerate(dense_results):
            cid = doc["chunk_id"]
            score_map[cid] += 1 / (rank + 1 + rrf_k)
            doc_map[cid] = doc  # Store full metadata

        # Process sparse results
        for rank, doc in enumerate(sparse_results):
            cid = doc["chunk_id"]
            score_map[cid] += 1 / (rank + 1 + rrf_k)
            doc_map[cid] = doc  # Will overwrite if duplicate, but that's fine

        # Sort by combined RRF score
        sorted_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Assemble final results
        final_results = []
        for cid, score in sorted_items:
            result = doc_map[cid].copy()
            result["rrf_score"] = score
            final_results.append(result)

        return final_results
    
    def check_db_populated(self, emb_model, emb_endpoint, max_tokens):
        self._ensure_embedder(emb_model, emb_endpoint, max_tokens)
        self.collection_name = self._generate_collection_name()

        if not utility.has_collection(self.collection_name):
            return False
        return True

    def search(self, query, emb_model, emb_endpoint, max_tokens, top_k=5, mode='', language='', subject='', class_level=''):
        self._ensure_embedder(emb_model, emb_endpoint, max_tokens)
        self.collection_name = self._generate_collection_name()

        if not utility.has_collection(self.collection_name):
            raise MilvusNotReadyError(
                    f"Milvus database is empty. Ingest documents first."
                )

        query_vector = self._embedder.embed_query(query)
        self.collection = Collection(name=self.collection_name)
        self.collection.load()

        if mode == "dense":
            results = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=top_k * 3,  # retrieve more for filtering
                output_fields=["chunk_id", "page_content", "filename", "type", "source", "language", "subject", "class_level", "page_numbers"],
                expr = build_filter_expr(language, subject, class_level)
            )
            dense_results = [hit.get('entity') for hit in results[0]]
            dense_results = dense_results[:top_k]
            
            return dense_results

        elif mode == "sparse":
            if self.bm25 is None:
                loaded = self._load_sparse_index()
                if not loaded:
                    raise RuntimeError("Sparse search index not initialized.")

            query_tokens = tokenize_for_bm25(query, language or "English")
            scores = self.bm25.get_scores(query_tokens)
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:3*top_k]

            sparse_results = []
            use_filters = any([language, subject, class_level])
            for idx, score in ranked:
                metadata = self.metadata_map[idx]
                if not use_filters or metadata_matches_filters(metadata, language, subject, class_level):
                    sparse_results.append({**metadata, "score": float(score)})
                if len(sparse_results) >= top_k:
                    break

            return sparse_results

        elif mode == "hybrid":
            if self.bm25 is None:
                loaded = self._load_sparse_index()
                if not loaded:
                    raise RuntimeError("Sparse index missing for hybrid search.")

            dense_results = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=top_k * 3,  # retrieve more for filtering
                output_fields=["chunk_id", "page_content", "filename", "type", "source", "language", "subject", "class_level", "page_numbers"],
                expr = build_filter_expr(language, subject, class_level)
            )
            dense_results = [hit.get('entity') for hit in dense_results[0]]
            dense_results = dense_results[:top_k]

            query_tokens = tokenize_for_bm25(query, language or "English")
            scores = self.bm25.get_scores(query_tokens)
            sparse_ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:3*top_k] # retrieve more for filtering

            sparse_results = []
            for idx, score in sparse_ranked:
                metadata = self.metadata_map[idx]
                use_filters = any([language, subject, class_level])
                if not use_filters or metadata_matches_filters(metadata, language, subject, class_level):
                    sparse_results.append({**metadata, "score": score})
                if len(sparse_results) >= top_k:
                    break

            return self._rrf_fusion(dense_results, sparse_results, top_k)

        else:
            raise ValueError("Invalid search mode. Choose from ['dense', 'sparse', 'hybrid'].")
