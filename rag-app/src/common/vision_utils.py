import logging
from typing import List, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm

from ..common.misc_utils import get_logger
from ..common.settings import get_settings

logger = get_logger("VLM")

is_debug = logger.isEnabledFor(logging.DEBUG)
tqdm_wrapper = tqdm if is_debug else (lambda x, **kwargs: x)

settings = get_settings()

SESSION: requests.Session | None = None


def create_vlm_session(
    pool_maxsize: int,
    pool_connections: int = 1,
    pool_block: bool = True,
) -> None:
    """Create a shared HTTP session for vLLM calls (if not already created).

    Parameters
    ----------
    pool_maxsize:
        Maximum number of connections in the pool.
    pool_connections:
        Number of connection pools to cache.
    pool_block:
        Whether the connection pool should block when no free connections are available.
    """
    global SESSION

    if SESSION is not None:
        return

    adapter = HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        pool_block=pool_block,
    )

    session = requests.Session()
    #session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.verify=False

    SESSION = session
    logger.debug(
        "Initialized VLM session with pool_maxsize=%d, pool_connections=%d, pool_block=%s",
        pool_maxsize,
        pool_connections,
        pool_block,
    )


def _vlm_chat_with_image(
    image_uri: str,
    prompt_text: str,
    model_id: str,
    vlm_endpoint: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Low-level helper to call the vLLM vision endpoint once.

    Assumes `create_vlm_session` has already been called.
    """
    if SESSION is None:  # Safety net; should normally be initialized by public APIs.
        create_vlm_session(pool_maxsize=4)

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        logger.debug(
            "Calling VLM at %s/v1/chat/completions with model=%s",
            vlm_endpoint,
            model_id,
        )
        response = SESSION.post(f"{vlm_endpoint}/v1/chat/completions", json=payload)  # type: ignore[arg-type]
        response.raise_for_status()
        result = response.json()
        content = (result.get("choices", [{}])[0]
                         .get("message", {})
                         .get("content", "") or "").strip()
        logger.debug("VLM raw response content: %s", content)
        return content
    except requests.exceptions.RequestException as e:
        error_details = str(e)
        if e.response is not None:
            error_details += f", Response Text: {e.response.text}"
        logger.error("Error calling vLLM vision API: %s", error_details)
        raise
    except Exception as e:  # noqa: BLE001
        logger.error("Unexpected error calling vLLM vision API: %s", e)
        raise


def _build_image_summary_prompt(caption: str | None = None) -> str:
    """Build the prompt used for image summarisation from settings.

    Parameters
    ----------
    caption:
        Optional caption associated with the image.

    Returns
    -------
    str
        The full prompt text.
    """
    prompt = settings.prompts.image_summary.strip()

    if caption:
        caption = caption.strip()
        if caption:
            prompt += f"\n\nThe caption of the image is:\n{caption}\n"

    return prompt


def generate_image_summary_helper(
    image_uri: str,
    caption: str,
    model_id: str,
    vlm_endpoint: str,
    retries: int = 3,
) -> str:
    """Generate a detailed summary for a single image.

    Parameters
    ----------
    image_uri:
        URI for the image. This can be an HTTP(S) URL or a data URI.
    caption:
        Optional caption text associated with the image.
    model_id:
        Name of the vision-capable model served by vLLM.
    vlm_endpoint:
        Base URL of the vLLM OpenAI-compatible endpoint.
    retries:
        Number of times to retry the call on failure.

    Returns
    -------
    str
        The generated image summary, or a fallback message on failure.
    """
    prompt_text = _build_image_summary_prompt(caption)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            logger.debug(
                "Summarising image (attempt %d/%d) with model '%s'",
                attempt,
                retries,
                model_id,
            )
            summary = _vlm_chat_with_image(
                image_uri=image_uri,
                prompt_text=prompt_text,
                model_id=model_id,
                vlm_endpoint=vlm_endpoint,
                max_tokens=512,
                temperature=0.0,
            )
            logger.debug("Image summary generated successfully.")
            return summary.strip()
        except Exception as e:  # noqa: BLE001
            last_error = e
            logger.warning(
                "[Attempt %d/%d] Error during image summary: %s",
                attempt,
                retries,
                e,
            )

    logger.error(
        "Image summary failed after %d attempts. Last error: %s",
        retries,
        last_error,
    )
    return "Image summary failed after multiple attempts."


def generate_image_summary(
    image_info_list: Sequence[Tuple[str, str]],
    vlm_model: str,
    vlm_endpoint: str,
    max_workers: int = 8,
    retries: int = 3,
) -> List[str]:
    """Generate summaries for a batch of images in parallel.

    Parameters
    ----------
    image_info_list:
        Sequence of (image_uri, caption) tuples.
    vlm_model:
        Name of the vision-capable model served by vLLM.
    vlm_endpoint:
        Base URL of the vLLM OpenAI-compatible endpoint.
    max_workers:
        Maximum number of worker threads for parallel processing.
    retries:
        Number of times to retry each image on failure.

    Returns
    -------
    List[str]
        A list of summaries aligned with ``image_info_list``.
    """
    if not image_info_list:
        return []

    max_workers = max(1, min(max_workers, len(image_info_list)))

    # Initialize a shared session with a pool sized for our concurrency.
    create_vlm_session(pool_maxsize=max_workers)

    def process_image(image_uri: str, caption: str) -> str:
        return generate_image_summary_helper(
            image_uri=image_uri,
            caption=caption or "",
            model_id=vlm_model,
            vlm_endpoint=vlm_endpoint,
            retries=retries,
        )

    summaries: List[str] = ["" for _ in range(len(image_info_list))]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_image, image_uri, caption): idx
            for idx, (image_uri, caption) in enumerate(image_info_list)
        }

        for future in tqdm_wrapper(
            as_completed(futures),
            total=len(futures),
            desc="Summarising images",
        ):
            idx = futures[future]
            try:
                summaries[idx] = future.result()
            except Exception as e:
                logger.error("Thread failed for image index %d: %s", idx, e)
                summaries[idx] = "Image summary failed."

    return summaries
