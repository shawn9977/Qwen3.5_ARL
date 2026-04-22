from transformers import AutoProcessor
from transformers.video_utils import load_video
from huggingface_hub import hf_hub_download
from optimum.intel.openvino import OVModelForVisualCausalLM

model_dir = "Qwen3.5-9B-INT4"

processor = AutoProcessor.from_pretrained(model_dir)
model = OVModelForVisualCausalLM.from_pretrained(model_dir)
device='GPU'
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
