
from huggingface_hub import login
login()


model = "facebook/dinov3-vitl16-pretrain-lvd1689m"
from huggingface_hub import hf_hub_download
hf_hub_download(model, "config.json")


