import json
import time
import logging
import os
import uuid
import base64
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import requests
from tqdm import tqdm
import pdfplumber
import base64
from io import BytesIO
os.environ['GRPC_VERBOSITY'] = 'ERROR' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path
from docling.datamodel.document import DoclingDocument
from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor
from threading import Semaphore
from indicnlp.tokenize.sentence_tokenize import sentence_split

from ..common.llm_utils import create_llm_session, classify_text_with_llm, summarize_table, tokenize_with_llm
from ..common.vision_utils import generate_image_summary
from ..common.misc_utils import get_logger, generate_file_checksum, text_suffix, table_suffix, image_suffix
from ..ingest.pdf_utils import get_toc, get_matching_header_lvl, load_pdf_pages, find_text_font_size


SETTINGS_PATH = Path(__file__).resolve().parent.parent / "settings.json"
with open(SETTINGS_PATH, 'r') as f:
    data = json.load(f)
docling_server_url = str(data.get("docling_server_url"))
docling_server_port = int(data.get("docling_server_port"))

PAGES_PER_CHUNK = 15
MAX_IN_FLIGHT_TASKS = 4
POLL_DELAY = 5.0
MAX_WAIT_SECONDS = 5 * 60 # 5 minutes
MAX_PARALLEL_CHUNKS = 8 # docling_workers

logging.getLogger('docling').setLevel(logging.CRITICAL)

logger = get_logger("Docling")

is_debug = logger.isEnabledFor(logging.DEBUG) 
tqdm_wrapper = None
if is_debug:
    tqdm_wrapper = tqdm
else:
    tqdm_wrapper = lambda x, **kwargs: x

excluded_labels = {
    'page_header', 'page_footer', 'caption', 'reference', 'footnote'
}

# Per processor pool 8 requests are getting spawned, since we are creating 4 processor pools, used 8 to match the vLLM's Max Batch Size 32
POOL_SIZE = 8
IMAGE_RESOLUTION_SCALE = 1.0

create_llm_session(pool_maxsize=POOL_SIZE)

def pil_to_data_uri(img) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def process_converted_document(res, pdf_path, out_path, gen_model, gen_endpoint, vlm_model, vlm_endpoint, start_time, timings):    
    doc_json = res.export_to_dict()
    stem = Path(pdf_path).stem

    # Initialize TocHeaders to get the Table of Contents (TOC)
    toc_headers = None
    page_count = 0
    try:
        toc_headers, page_count = get_toc(pdf_path)
    except Exception as e:
        logger.debug(f"No TOC found or failed to load TOC: {e}")

    # Load pdf pages one time when TOC headers not found for retrieving the font size of header texts
    pdf_pages = None
    if not toc_headers:
        pdf_pages = load_pdf_pages(pdf_path)
        page_count = len(pdf_pages)

    # --- Text Extraction ---
    t0 = time.time()
    filtered_blocks, table_captions, image_captions = [], [], []
    for block in doc_json.get('texts', []):
        block_type = block.get('label', '')
        if block_type not in excluded_labels:
            filtered_blocks.append(block)
        if block_type == 'caption':
            block_parent = block.get('parent', {}).get('$ref', '')
            if 'tables' in block_parent:
                table_captions.append(block)
            elif 'pictures' in block_parent:
                image_captions.append(block)
    timings['extract_text_blocks'] = time.time() - t0

    if len(filtered_blocks):

        filtered_text_dicts = filtered_blocks

        structured_output = []
        last_header_level = 0
        t0 = time.time()
        for text_obj in tqdm_wrapper(filtered_text_dicts, desc=f"Processing text content of '{pdf_path}'"):
            label = text_obj.get("label", "")

            # Check if it's a section header and process TOC or fallback to font size extraction
            if label == "section_header":
                prov_list = text_obj.get("prov", [])

                for prov in prov_list:
                    page_no = prov.get("page_no")
                    bbox_dict = prov.get("bbox")

                    if page_no is None or bbox_dict is None:
                        continue

                    if toc_headers:
                        header_prefix = get_matching_header_lvl(toc_headers, text_obj.get("text", ""))
                        if header_prefix:
                            # If TOC matches, use the level from TOC
                            structured_output.append({
                                "label": label,
                                "text": f"{header_prefix} {text_obj.get('text', '')}",
                                "page": page_no,
                                "font_size": None,  # Font size isn't necessary if TOC matches
                            })
                            last_header_level = len(header_prefix.strip())  # Update last header level
                        else:
                            # If no match, use the previous header level + 1
                            new_header_level = last_header_level + 1
                            structured_output.append({
                                "label": label,
                                "text": f"{'#' * new_header_level} {text_obj.get('text', '')}",
                                "page": page_no,
                                "font_size": None,  # Font size isn't necessary if TOC matches
                            })
                    else:
                        matches = find_text_font_size(pdf_pages, text_obj.get("text", ""), page_no - 1)
                        if len(matches):
                            font_size = 0
                            count = 0
                            for match in matches:
                                font_size += match["font_size"] if match["match_score"] == 100 else 0
                                count += 1 if match["match_score"] == 100 else 0
                            font_size = font_size / count if count else None

                            structured_output.append({
                                "label": label,
                                "text": text_obj.get("text", ""),
                                "page": page_no,
                                "font_size": round(font_size, 2) if font_size else None
                            })
            else:
                structured_output.append({
                    "label": label,
                    "text": text_obj.get("text", ""),
                    "page": text_obj.get("prov")[0].get("page_no"),
                    "font_size": None
                })

        timings["font_size_extraction"] = time.time() - t0

        (Path(out_path) / f"{stem}{text_suffix}").write_text(json.dumps(structured_output, indent=2), encoding="utf-8")
        
    else:
        (Path(out_path) / f"{stem}{text_suffix}").write_text(json.dumps(filtered_blocks, indent=2), encoding="utf-8")

    # --- Table Extraction ---
    table_count = len(res.tables)
    if table_count:
        t0 = time.time()
        table_htmls_dict = {}
        table_captions_dict = {i: None for i in range(len(res.tables))}
        for table_ix, table in enumerate(tqdm_wrapper(res.tables, desc=f"Processing table content of '{pdf_path}'")):
            table_htmls_dict[table_ix] = table.export_to_html(doc=res)
            for caption_idx, block in enumerate(table_captions):
                if block.get('parent')['$ref'] == f'#/tables/{table_ix}':
                    table_captions_dict[table_ix] = block.get('text', '')
                    table_captions.pop(caption_idx)
                    break
        table_htmls = [table_htmls_dict[key] for key in sorted(table_htmls_dict)]
        table_captions_list = [table_captions_dict[key] for key in sorted(table_captions_dict)]
        timings['extract_tables'] = time.time() - t0

        t0 = time.time()
        table_summaries = summarize_table(table_htmls, gen_model, gen_endpoint, pdf_path)
        timings['summarize_tables'] = time.time() - t0

        t0 = time.time()
        decisions = classify_text_with_llm(table_summaries, gen_model, gen_endpoint, pdf_path, "table")
        filtered_table_dicts = {
            idx: {
                'html': html,
                'caption': caption,
                'summary': summary
            }
            for idx, (keep, html, caption, summary) in enumerate(zip(decisions, table_htmls, table_captions_list, table_summaries)) if keep
        }
        (Path(out_path) / f"{stem}{table_suffix}").write_text(json.dumps(filtered_table_dicts, indent=2), encoding="utf-8")
        timings['filter_tables'] = time.time() - t0
    else:
        (Path(out_path) / f"{stem}{table_suffix}").write_text(json.dumps([], indent=2), encoding="utf-8")

    # --- Image Extraction ---
    image_count = len(doc_json.get('pictures', []))
    if image_count:
        t0 = time.time()
        image_dict, image_uris, ordered_image_captions = [], [], []

        picture_items = [
            item for item, _ in res.iterate_items() if item.label == "picture"
        ]

        for image_idx, (block, picture_item) in enumerate(
            zip(doc_json.get('pictures', []), picture_items)
        ):
            caption = ''
            for child in block.get('children', []):
                child_id = child['$ref']
                for caption_idx, child_block in enumerate(image_captions):
                    if child_block.get('self_ref', '') == child_id:
                        caption += f'{child_block["text"]} '
                        image_captions.pop(caption_idx)
                        break

            img = picture_item.get_image(doc=res)
            uri = pil_to_data_uri(img) if img is not None else ""

            image_uris.append(uri)
            ordered_image_captions.append(caption)

        timings['extract_images'] = time.time() - t0

        t0 = time.time()
        image_summaries = generate_image_summary(
            list(zip(image_uris, ordered_image_captions)),
            vlm_model,
            vlm_endpoint,
        )
        timings['generate_image_summaries'] = time.time() - t0

        t0 = time.time()
        for idx, (uri, summary, caption) in enumerate(
            zip(image_uris, image_summaries, ordered_image_captions)
        ):
            image_dict.append(
                {idx: {'image': uri, 'caption': caption, 'summary': summary}}
            )

        decisions = classify_text_with_llm(
            image_summaries, gen_model, gen_endpoint, pdf_path, "image"
        )
        filtered_image_dicts = [
            image_dict[idx] for idx, keep in enumerate(decisions) if keep
        ]

        (Path(out_path) / f"{stem}{image_suffix}").write_text(json.dumps(filtered_image_dicts, indent=2), encoding="utf-8")
        timings['filter_image_summaries'] = time.time() - t0
    else:
        (Path(out_path) / f"{stem}{image_suffix}").write_text(json.dumps([], indent=2), encoding="utf-8")


    total_time = time.time() - start_time
    logger.debug(f"Timing for {stem} Total: {total_time:.2f}s")
    for k, v in timings.items():
        logger.debug(f"  {k:<30}: {v:.2f}s")
    return page_count, table_count, image_count

def post_file_async(
    file_path: Path,
    start_page: int,
    end_page: int,
    do_ocr: bool = False,
    max_retries: int = 3,
) -> str:
    data = {
        "to_formats": ["json"],
        "do_ocr": do_ocr,
        "force_ocr": do_ocr,
        "include_images": True,
        "images_scale": 1.,
        "image_export_mode": "embedded",
        "table_mode": "accurate",
        "do_table_structure": True,
        "abort_on_error": False,
        "page_range": [start_page, end_page],
        "ocr_engine": "easyocr",
        "ocr_lang": ["hi", "en"],
    }

    for attempt in range(1, max_retries + 1):
        try:
            with file_path.open("rb") as f:
                response = requests.post(
                    f"{docling_server_url}:{docling_server_port}/v1/convert/file/async",
                    files={"files": (file_path.name, f, "application/pdf")},
                    data=data,
                    timeout=30,
                    verify=False 
                )

            if response.status_code == 200:
                return response.json()["task_id"]

            raise RuntimeError(response.text)

        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Docling async submit failed after {max_retries} attempts: {e}"
                )
            time.sleep(2 ** attempt)

def poll_task(task_id: str) -> dict:
    start_time = time.time()

    while True:
        response = requests.get(
            f"{docling_server_url}:{docling_server_port}/v1/result/{task_id}",
            timeout=30,
            verify=False 
        )

        if response.status_code == 200:
            return response.json()["document"]["json_content"]

        if response.status_code != 404:
            raise RuntimeError(
                f"Docling task {task_id} failed "
                f"({response.status_code}): {response.text}"
            )

        elapsed = time.time() - start_time
        if elapsed > MAX_WAIT_SECONDS:
            raise TimeoutError(
                f"Docling task {task_id} exceeded max wait time"
            )
        time.sleep(POLL_DELAY)

def _process_chunk(pdf_path: Path, start: int, end: int, do_ocr: bool) -> DoclingDocument:
    task_id = post_file_async(pdf_path, start, end, do_ocr=do_ocr)
    doc_json = poll_task(task_id)
    return DoclingDocument.model_validate(doc_json)

# def convert_via_docling_server(path: str, do_ocr: bool = False) -> dict:
#     pdf_path = Path(path)

#     # Count pages safely
#     with pdfplumber.open(pdf_path) as pdf:
#         total_pages = len(pdf.pages)

#     # Build chunk ranges
#     chunks = [
#         (start, min(start + PAGES_PER_CHUNK - 1, total_pages))
#         for start in range(1, total_pages + 1, PAGES_PER_CHUNK)
#     ]

#     results: dict[int, DoclingDocument] = {}

#     with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CHUNKS) as executor:
#         future_map = {
#             executor.submit(_process_chunk, pdf_path, start, end, do_ocr): start
#             for start, end in chunks
#         }

#         for future in as_completed(future_map):
#             start_page = future_map[future]
#             try:
#                 results[start_page] = future.result()
#             except Exception as e:
#                 raise RuntimeError(
#                     f"Failed processing pages starting at {start_page}: {e}"
#                 )

#     # Preserve page order
#     ordered_docs = [results[start] for start, _ in sorted(chunks)]

#     return DoclingDocument.concatenate(ordered_docs).export_to_dict()

def convert_via_docling_server(path: str, do_ocr: bool = False) -> dict:
    pdf_path = Path(path)

    # ---------- Step 1: Count pages safely ----------
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
    except Exception as e:
        print(f"[Docling] Failed to read PDF pages for {path}: {e}")
        return {}  # return empty document so pipeline continues

    if total_pages == 0:
        print(f"[Docling] No pages found in {path}")
        return {}

    # ---------- Step 2: Build chunk ranges ----------
    chunks = [
        (start, min(start + PAGES_PER_CHUNK - 1, total_pages))
        for start in range(1, total_pages + 1, PAGES_PER_CHUNK)
    ]

    results: dict[int, DoclingDocument] = {}
    failed_chunks = []

    # ---------- Step 3: Process chunks in parallel ----------
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CHUNKS) as executor:
        future_map = {
            executor.submit(_process_chunk, pdf_path, start, end, do_ocr): start
            for start, end in chunks
        }

        for future in as_completed(future_map):
            start_page = future_map[future]
            try:
                results[start_page] = future.result()
            except Exception as e:
                print(f"[Docling] Chunk {start_page} failed for {path}: {e}")
                failed_chunks.append(start_page)

    # ---------- Step 4: Handle total failure ----------
    if not results:
        print(f"[Docling] All chunks failed for {path}")
        return {}

    # ---------- Step 5: Preserve order, skip failed chunks ----------
    ordered_docs = []
    for start, _ in sorted(chunks):
        if start in results:
            ordered_docs.append(results[start])

    # ---------- Step 6: Concatenate safely ----------
    try:
        combined = DoclingDocument.concatenate(ordered_docs)
        return combined.export_to_dict()
    except Exception as e:
        print(f"[Docling] Concatenation failed for {path}: {e}")
        # Fallback: return first successful chunk instead of crashing
        try:
            return ordered_docs[0].export_to_dict()
        except Exception:
            return {}

def convert_and_process(
    path,
    out_path,
    llm_model,
    llm_endpoint,
    vlm_model,
    vlm_endpoint
):
    try:
        logger.info(f"Processing '{path}'")
        timings = {}
        start_time = time.time()
        f = (Path(out_path) / f"{Path(path).stem}_converted.json")
        logger.debug(f"Checking {str(f)}")
        converted_doc = None

        if f.exists():
            logger.debug("Loading from converted json")
            with Path(str(f)).open("r") as fp:
                doc_dict = json.load(fp)
                converted_doc = DoclingDocument.model_validate(doc_dict)

        else:
            logger.debug(f"Not exist, converting '{path}'")
            t0 = time.time()
            language = Path(path).parts[-4]  # Language folder
            do_ocr = language.lower() != "english"
            doc_dict = convert_via_docling_server(path, do_ocr=do_ocr)
            converted_doc = DoclingDocument.model_validate(doc_dict)

            timings["conversion_time"] = time.time() - t0
            logger.debug(f"'{path}' converted")

            converted_doc.save_as_json(str(f))

        page_count, table_count, image_count = process_converted_document(
            converted_doc,
            path,
            out_path,
            llm_model,
            llm_endpoint,
            vlm_model,
            vlm_endpoint,
            start_time,
            timings,
        )

        return path, {
            "page_count": page_count,
            "table_count": table_count,
            "image_count": image_count,
        }

    except Exception as e:
        raise Exception(f"Error converting and processing '{path}': {e}")


def extract_document_data(
    input_paths,
    out_path,
    llm_model,
    llm_endpoint,
    vlm_model,
    vlm_endpoint,
    force=False,
):
    # ------------------------------------------------------------------
    # Skip files that already exist by matching checksum
    # ------------------------------------------------------------------

    filtered_input_paths = []
    converted_paths = []

    for path in input_paths:
        checksum_file = Path(out_path) / f"{Path(path).stem}.checksum"

        if checksum_file.exists() and not force:
            checksum = checksum_file.read_text()
            if (
                checksum == generate_file_checksum(path)
                and (Path(out_path) / f"{Path(path).stem}{text_suffix}").exists()
                and (Path(out_path) / f"{Path(path).stem}{table_suffix}").exists()
                and (Path(out_path) / f"{Path(path).stem}{image_suffix}").exists()
            ):
                converted_paths.append(path)
                continue

        filtered_input_paths.append(path)

    # Write checksums
    for path in filtered_input_paths:
        checksum = generate_file_checksum(path)
        (Path(out_path) / f"{Path(path).stem}.checksum").write_text(
            checksum, encoding="utf-8"
        )

    converted_pdf_stats = {}

    if not filtered_input_paths:
        logger.debug("No files to convert and process")
        return converted_paths, converted_pdf_stats

    for path in filtered_input_paths:
        try:
            path, pdf_stats = convert_and_process(
                path,
                out_path,
                llm_model,
                llm_endpoint,
                vlm_model,
                vlm_endpoint,
            )
            converted_paths.append(path)
            converted_pdf_stats[path] = pdf_stats
            logger.info(f"Processed '{path}'")
        except Exception as e:
            logger.error(f"{e}")

    return converted_paths, converted_pdf_stats


def collect_header_font_sizes(elements):
    """
    elements: list of dicts with at least keys: 'label', 'font_size'
    Returns a sorted list of unique section_header font sizes, descending.
    """
    sizes = {
        el['font_size']
        for el in elements
        if el.get('label') == 'section_header' and el.get('font_size') is not None
    }
    return sorted(sizes, reverse=True)

def get_header_level(text, font_size, sorted_font_sizes):
    """
    Determine header level based on markdown syntax or font size hierarchy.
    """
    text = text.strip()

    # Priority 1: Markdown syntax
    if text.startswith('#'):
        level = len(text.strip()) - len(text.strip().lstrip('#'))
        return level, text.strip().lstrip('#').strip()

    # Priority 2: Font size ranking
    try:
        level = sorted_font_sizes.index(font_size) + 1
    except ValueError:
        # Unknown font size; assign lowest priority
        level = len(sorted_font_sizes)

    return level, text


def count_tokens(text, emb_endpoint):
    token_len = len(tokenize_with_llm(text, emb_endpoint))
    return token_len

def split_text_into_token_chunks(text, emb_endpoint, language, max_tokens=512, overlap=50):
    sentences = sentence_split(text, lang=language)
    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        token_len = count_tokens(sentence, emb_endpoint)

        if current_token_count + token_len > max_tokens:
            # save current chunk
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            # overlap logic (optional)
            if overlap > 0 and len(current_chunk) > 0:
                overlap_text = current_chunk[-1]
                current_chunk = [overlap_text]
                current_token_count = count_tokens(overlap_text, emb_endpoint)
            else:
                current_chunk = []
                current_token_count = 0

        current_chunk.append(sentence)
        current_token_count += token_len

    # flush last
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(chunk_text)

    return chunks


def flush_chunk(current_chunk, chunks, language, emb_endpoint, max_tokens):
    content = current_chunk["content"].strip()
    if not content:
        return

    # Split content into token chunks
    token_chunks = split_text_into_token_chunks(content, emb_endpoint, language, max_tokens=max_tokens)

    for i, part in enumerate(token_chunks):
        chunk = {
            "chapter_title": current_chunk["chapter_title"],
            "section_title": current_chunk["section_title"],
            "subsection_title": current_chunk["subsection_title"],
            "subsubsection_title": current_chunk["subsubsection_title"],
            "content": part,
            "page_range": sorted(set(current_chunk["page_range"])),
            "source_nodes": current_chunk["source_nodes"].copy()
        }
        if len(token_chunks) > 1:
            chunk["part_id"] = i + 1
        chunks.append(chunk)

    # Reset current_chunk after flushing
    current_chunk["chapter_title"] = ""
    current_chunk["section_title"] = ""
    current_chunk["subsection_title"] = ""
    current_chunk["subsubsection_title"] = ""
    current_chunk["content"] = ""
    current_chunk["page_range"] = []
    current_chunk["source_nodes"] = []


def chunk_single_file(input_path, output_path, language, emb_endpoint, max_tokens=512):    
    if not Path(output_path).exists():
        with open(input_path, "r") as f:
            data = json.load(f)
        
        font_size_levels = collect_header_font_sizes(data)

        chunks = []
        current_chunk = {
            "chapter_title": None,
            "section_title": None,
            "subsection_title": None,
            "subsubsection_title": None,
            "content": "",
            "page_range": [],
            "source_nodes": []
        }

        current_chapter = None
        current_section = None
        current_subsection = None
        current_subsubsection = None

        for idx, block in enumerate(tqdm_wrapper(data, desc=f"Chunking {input_path}")):
            label = block.get("label")
            text = block.get("text", "").strip()
            try:
                page_no = block.get("prov", {})[0].get("page_no")
            except:
                page_no = 0
            ref = f"#texts/{idx}"

            if label == "section_header":
                level, full_title = get_header_level(text, block.get("font_size"), font_size_levels)
                if level == 1:
                    current_chapter = full_title
                    current_section = None
                    current_subsection = None
                    current_subsubsection = None
                elif level == 2:
                    current_section = full_title
                    current_subsection = None
                    current_subsubsection = None
                elif level == 3:
                    current_subsection = full_title
                    current_subsubsection = None
                else:
                    current_subsubsection = full_title

                # Flush current chunk and update
                flush_chunk(current_chunk, chunks, language, emb_endpoint, max_tokens)
                current_chunk["chapter_title"] = current_chapter
                current_chunk["section_title"] = current_section
                current_chunk["subsection_title"] = current_subsection
                current_chunk["subsubsection_title"] = current_subsubsection

            elif label in {"text", "list_item", "code", "formula"}:
                if current_chunk["chapter_title"] is None:
                    current_chunk["chapter_title"] = current_chapter
                if current_chunk["section_title"] is None:
                    current_chunk["section_title"] = current_section
                if current_chunk["subsection_title"] is None:
                    current_chunk["subsection_title"] = current_subsection
                if current_chunk["subsubsection_title"] is None:
                    current_chunk["subsubsection_title"] = current_subsubsection

                if label == 'code':
                    current_chunk["content"] += f"```\n{text}\n``` "
                elif label == 'formula':
                    current_chunk["content"] += f"${text}$ "
                else:
                    current_chunk["content"] += f"{text} "
                if page_no is not None:
                    current_chunk["page_range"].append(page_no)
                current_chunk["source_nodes"].append(ref)
            else:
                logger.debug(f'Skipping adding "{label}".')

        # Flush any remaining content
        flush_chunk(current_chunk, chunks, language, emb_endpoint, max_tokens)

        # Save the processed chunks to the output file
        with open(output_path, "w") as f:
            json.dump(chunks, f, indent=2)

        logger.debug(f"{len(chunks)} RAG chunks saved to {output_path}")
    else:
        logger.debug(f"{output_path} already exists.")

    return output_path

def hierarchical_chunk_with_token_split(input_paths, output_paths, languages, emb_endpoint, max_tokens=512):
    if len(input_paths) != len(output_paths):
        raise ValueError("`input_paths` and `output_paths` must have the same length")

    # Process each input-output file pair in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for input_path, output_path, language in zip(input_paths, output_paths, languages):
            logger.debug(f"Submitting '{input_path}' for chunking")
            futures.append(executor.submit(chunk_single_file, input_path, output_path, language, emb_endpoint, max_tokens))

        chunked_files = []
        # Wait for all futures to finish and handle exceptions
        for future in tqdm_wrapper(futures, desc="Creating chunks from processed documents"):
            try:
                chunked_files.append(future.result())  # Capture exceptions if any
            except Exception as e:
                logger.error(f"Error occurred while chunking: {e}")
    logger.debug("Chunks creation completed")
    return chunked_files

def create_chunk_documents(in_txt_f, in_tab_f, in_img_f, orig_fn, out_path, collection_name, metadata):
    language = metadata.get("language", "")
    subject = metadata.get("subject", "")
    class_level = metadata.get("class_level", "")

    logger.debug(f"Creating combined chunk documents from '{in_txt_f}', '{in_tab_f}', and '{in_img_f}'")

    os.makedirs(f'{out_path}/images_{collection_name}', exist_ok=True)

    with open(in_txt_f, "r") as f:
        txt_data = json.load(f)

    with open(in_tab_f, "r") as f:
        tab_data = json.load(f)

    with open(in_img_f, "r") as f:
        img_data = json.load(f)

    txt_docs = []
    if len(txt_data):
        for _, block in enumerate(txt_data):
            meta_info = ''
            if block.get('chapter_title'):
                meta_info += f"Chapter: {block.get('chapter_title')} "
            if block.get('section_title'):
                meta_info += f"Section: {block.get('section_title')} "
            if block.get('subsection_title'):
                meta_info += f"Subsection: {block.get('subsection_title')} "
            if block.get('subsubsection_title'):
                meta_info += f"Subsubsection: {block.get('subsubsection_title')} "
            txt_docs.append({
                "page_content": f'{meta_info}\n{block.get("content")}' if meta_info != '' else block.get("content"),
                "filename": orig_fn,
                "type": "text",
                "source": meta_info,
                "language": language,
                "subject": subject,
                "class_level": class_level
            })

    tab_docs = []
    if len(tab_data):
        tab_data = list(tab_data.values())
        for tab_id, block in enumerate(tab_data):
            tab_docs.append({
                "page_content": block.get("summary"),
                "filename": orig_fn,
                "type": "table",
                "source": block.get("html"),
                "language": language,
                "subject": subject,
                "class_level": class_level
            })

    img_docs = []
    if len(img_data):
        for img_id, block in enumerate(img_data):
            block = list(block.values())[0]
            img_path = f"{out_path}/images_{collection_name}/{uuid.uuid5(uuid.uuid5(uuid.NAMESPACE_DNS, collection_name), f'{orig_fn.strip().lower()}_img_{str(img_id).strip()}').hex}.png"

            uri = block.get('image')
            # Split off the header if present
            if ',' in uri:
                _, b64_data = uri.split(',', 1)
            else:
                b64_data = uri
            try:
                # Decode base64 string safely
                img_data = base64.b64decode(b64_data)
                # Load the image using PIL
                image = Image.open(BytesIO(img_data))
                image.load()  # Ensure the image is fully loaded
                image.save(img_path)
                img_docs.append({
                    "page_content": block.get("summary"),
                    "filename": orig_fn,
                    "type": "image",
                    "source": img_path,
                    "language": language,
                    "subject": subject,
                    "class_level": class_level
                })
            except base64.binascii.Error:
                print("❌ Error: The base64 data is invalid.")
            except UnidentifiedImageError:
                print("❌ Error: Cannot identify image file. The data might not be a valid image.")
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
    
    combined_docs = txt_docs + tab_docs + img_docs

    logger.debug(f"Combined chunk documents created")

    return combined_docs
