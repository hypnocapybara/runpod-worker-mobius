from huggingface_hub import snapshot_download


def fetch_pretrained_model(model_name):
    """
    Fetches a pretrained model from the HuggingFace model hub.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return snapshot_download(repo_id=model_name, repo_type="model")
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise


def warm_up_pipeline():
    """
    Fetches the pipelines from the HuggingFace model hub.
    """

    fetch_pretrained_model("stabilityai/stable-diffusion-3-medium-diffusers")


if __name__ == "__main__":
    warm_up_pipeline()
