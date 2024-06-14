from huggingface_hub import hf_hub_download, login
import config

login(token=config.loginToken)
# Comments as I understand them lol
# Basic model no text
hf_hub_download(repo_id="stabilityai/stable-diffusion-3-medium", filename="sd3_medium.safetensors", local_dir="models")

# All in one but lower resource load
hf_hub_download(repo_id="stabilityai/stable-diffusion-3-medium", filename="sd3_medium_incl_clips.safetensors", local_dir="models")

# As above but more resource intensive
hf_hub_download(repo_id="stabilityai/stable-diffusion-3-medium", filename="sd3_medium_incl_clips_t5xxlfp8.safetensors", local_dir="models")

# Full model most resource intensive
hf_hub_download(repo_id="stabilityai/stable-diffusion-3-medium", filename="sd3_medium_incl_clips_t5xxlfp16.safetensors", local_dir="models")
