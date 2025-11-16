from huggingface_hub import hf_hub_download

from src.logger import logger


def fetch_data(repo_id: str, weights_path: str):
    try:
        logger.info(
            f"Attempting to download weights ({weights_path}) for model `{repo_id}`"
        )
        file_path = hf_hub_download(repo_id=repo_id, filename=weights_path)
    except Exception as e:
        logger.error(f"An error occured while downloading weigths: {e}")
    logger.info(f"Model weights have been saved to {file_path}")
