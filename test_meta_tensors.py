from transformers import AutoModelForImageSegmentation

m = AutoModelForImageSegmentation.from_pretrained(
    "briaai/RMBG-2.0",
    trust_remote_code=True,
    device_map=None,
    low_cpu_mem_usage=False,
    torch_dtype="float32",
)
print("OK:", sum(p.numel() for p in m.parameters()) > 0)

