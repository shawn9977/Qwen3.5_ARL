from transformers import AutoProcessor
from transformers.video_utils import load_video
from huggingface_hub import hf_hub_download
from optimum.intel.openvino import OVModelForVisualCausalLM

model_dir = "/home/intel/project/qwen35/Qwen3.5-35B-A3B/INT4"
device = "GPU"  # Force GPU with dynamic shape fix

processor = AutoProcessor.from_pretrained(model_dir)
# Disable CACHE_DIR to avoid cache export bug with dynamic shapes
model = OVModelForVisualCausalLM.from_pretrained(model_dir, device=device, ov_config={"CACHE_DIR": ""}, compile=False)


def read_execution_devices(obj):
    if obj is None:
        return None
    try:
        compiled = obj.get_compiled_model() if hasattr(obj, "get_compiled_model") else obj
        return compiled.get_property("EXECUTION_DEVICES")
    except Exception:
        return None


print(f"Compiling model on device: {device}")
try:
    model.compile()
    print(f"✓ Model compiled successfully on {device}")
except RuntimeError as exc:
    print(f"✗ Compilation error: {exc}")
    if device.upper() == "GPU":
        fallback_device = "AUTO:GPU,CPU"
        print(f"\nRetrying with {fallback_device}...")
        model = OVModelForVisualCausalLM.from_pretrained(model_dir, device=fallback_device, compile=False)
        model.compile()
        device = fallback_device
        print(f"✓ Model compiled successfully on {device}")
    else:
        raise

print(f"Configured device: {device}")
print("Execution devices:")
for comp_name in ["language_model", "vision_embeddings", "vision_embeddings_merger", "vision_embeddings_pos"]:
    comp = getattr(model, comp_name, None)
    req = getattr(comp, "request", None) if comp is not None else None
    devices = read_execution_devices(req)
    if devices is not None:
        print(f"  - {comp_name}: {devices}")

# Prepare video input
video_path = hf_hub_download(
                repo_id="raushan-testing-hf/videos-test",
                filename="sample_demo_1.mp4",
                repo_type="dataset",
            )
input_video, _ = load_video(video_path, num_frames=10, backend="opencv")

messages = [
    {"role": "user", "content": [
        {"type": "video"},
        {"type": "text", "text": "Why is this video funny?"},
    ]}
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], videos=[input_video], return_tensors="pt")

# Run inference
output_ids = model.generate(**inputs, max_new_tokens=100)
output_text = processor.decode(output_ids[0], skip_special_tokens=True)

print(output_text)