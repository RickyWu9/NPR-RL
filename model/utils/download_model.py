"""
Download models from Hugging Face Hub
"""
import os
from huggingface_hub import snapshot_download

codet5_model_name = "Salesforce/codet5-base"
qwen25_coder_3b_model_name = "Qwen/Qwen2.5-Coder-3B"
def download_llm_model(model_name: str = None, save_directory: str = None):
    model_short_name = model_name.split('/')[-1]
    if save_directory is None:
        # Default to the assets directory in the project
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(f"Project root: {project_root}")
        save_directory = os.path.join(project_root, 'assets', model_short_name)
    
    # Create the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)
    
    print(f"Downloading model to {save_directory}")
    
    # Download the model
    model_dir = snapshot_download(
        repo_id=model_name,
        local_dir=save_directory,
        max_workers=2,
        endpoint = "https://hf-mirror.com",
    )
    
    print(f"Model downloaded successfully to {model_dir}")
    return model_dir

if __name__ == "__main__":
    # Download the model to the default location
    download_llm_model(model_name=codet5_model_name, save_directory="D://LLM//CodeT5")