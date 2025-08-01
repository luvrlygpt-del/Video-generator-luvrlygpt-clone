# PyTorch 2.8 (temporary hack)
import os
os.system('pip install --upgrade --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu126 "torch<2.9" spaces')

# Actual demo code
import spaces
import torch
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils.export_utils import export_to_video
import gradio as gr
import tempfile
import numpy as np
from PIL import Image
import random
import gc
from optimization import optimize_pipeline_


MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

LANDSCAPE_WIDTH = 832
LANDSCAPE_HEIGHT = 480
MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 81

MIN_DURATION = round(MIN_FRAMES_MODEL/FIXED_FPS,1)
MAX_DURATION = round(MAX_FRAMES_MODEL/FIXED_FPS,1)


pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID,
    transformer=WanTransformer3DModel.from_pretrained('cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers',
        subfolder='transformer',
        torch_dtype=torch.bfloat16,
        device_map='cuda',
    ),
    transformer_2=WanTransformer3DModel.from_pretrained('cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers',
        subfolder='transformer_2',
        torch_dtype=torch.bfloat16,
        device_map='cuda',
    ),
    torch_dtype=torch.bfloat16,
).to('cuda')

# load, fuse, unload before compilation
# pipe.load_lora_weights(
#    "vrgamedevgirl84/Wan14BT2VFusioniX", 
#    weight_name="FusionX_LoRa/Phantom_Wan_14B_FusionX_LoRA.safetensors", 
#     adapter_name="phantom"
# )

# pipe.set_adapters(["phantom"], adapter_weights=[0.95])
# pipe.fuse_lora(adapter_names=["phantom"], lora_scale=1.0)
# pipe.unload_lora_weights()


pipe.load_lora_weights(
   "vrgamedevgirl84/Wan14BT2VFusioniX", 
   weight_name="FusionX_LoRa/Phantom_Wan_14B_FusionX_LoRA.safetensors", 
    adapter_name="phantom"
)
kwargs = {}
kwargs["load_into_transformer_2"] = True
pipe.load_lora_weights(
   "vrgamedevgirl84/Wan14BT2VFusioniX", 
   weight_name="FusionX_LoRa/Phantom_Wan_14B_FusionX_LoRA.safetensors", 
    adapter_name="phantom_2", **kwargs
)
pipe.set_adapters(["phantom", "phantom_2"], adapter_weights=[1., 1.])
pipe.fuse_lora(adapter_names=["phantom"], lora_scale=3., components=["transformer"])
pipe.fuse_lora(adapter_names=["phantom_2"], lora_scale=1., components=["transformer_2"])
pipe.unload_lora_weights()

for i in range(2): 
    gc.collect()
    torch.cuda.synchronize() 
    torch.cuda.empty_cache()

optimize_pipeline_(pipe,
    image=Image.new('RGB', (LANDSCAPE_WIDTH, LANDSCAPE_HEIGHT)),
    prompt='prompt',
    height=LANDSCAPE_HEIGHT,
    width=LANDSCAPE_WIDTH,
    num_frames=MAX_FRAMES_MODEL,
)


default_prompt_i2v = "make this image come alive, cinematic motion, smooth animation"
default_negative_prompt = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"


def resize_image(image: Image.Image) -> Image.Image:
    if image.height > image.width:
        transposed = image.transpose(Image.Transpose.ROTATE_90)
        resized = resize_image_landscape(transposed)
        return resized.transpose(Image.Transpose.ROTATE_270)
    return resize_image_landscape(image)


def resize_image_landscape(image: Image.Image) -> Image.Image:
    target_aspect = LANDSCAPE_WIDTH / LANDSCAPE_HEIGHT
    width, height = image.size
    in_aspect = width / height
    if in_aspect > target_aspect:
        new_width = round(height * target_aspect)
        left = (width - new_width) // 2
        image = image.crop((left, 0, left + new_width, height))
    else:
        new_height = round(width / target_aspect)
        top = (height - new_height) // 2
        image = image.crop((0, top, width, top + new_height))
    return image.resize((LANDSCAPE_WIDTH, LANDSCAPE_HEIGHT), Image.LANCZOS)

def get_duration(
    input_image,
    prompt,
    negative_prompt,
    duration_seconds,
    guidance_scale,
    guidance_scale_2,
    steps,
    seed,
    randomize_seed,
    progress,
):
    return steps * 15

@spaces.GPU(duration=get_duration)
def generate_video(
    input_image,
    prompt,
    negative_prompt=default_negative_prompt,
    duration_seconds = MAX_DURATION,
    guidance_scale = 2.5,
    guidance_scale_2 = 3.5,
    steps = 6,
    seed = 42,
    randomize_seed = False,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generate a video from an input image using the Wan 2.1 I2V model with CausVid LoRA.
    
    This function takes an input image and generates a video animation based on the provided
    prompt and parameters. It uses the Wan 2.1 14B Image-to-Video model with CausVid LoRA
    for fast generation in 4-8 steps.
    
    Args:
        input_image (PIL.Image): The input image to animate. Will be resized to target dimensions.
        prompt (str): Text prompt describing the desired animation or motion.
        negative_prompt (str, optional): Negative prompt to avoid unwanted elements. 
            Defaults to default_negative_prompt (contains unwanted visual artifacts).
        duration_seconds (float, optional): Duration of the generated video in seconds.
            Defaults to 2. Clamped between MIN_FRAMES_MODEL/FIXED_FPS and MAX_FRAMES_MODEL/FIXED_FPS.
        guidance_scale (float, optional): Controls adherence to the prompt. Higher values = more adherence.
            Defaults to 1.0. Range: 0.0-20.0.
        guidance_scale_2 (float, optional): Controls adherence to the prompt. Higher values = more adherence.
            Defaults to 1.0. Range: 0.0-20.0.
        steps (int, optional): Number of inference steps. More steps = higher quality but slower.
            Defaults to 4. Range: 1-30.
        seed (int, optional): Random seed for reproducible results. Defaults to 42.
            Range: 0 to MAX_SEED (2147483647).
        randomize_seed (bool, optional): Whether to use a random seed instead of the provided seed.
            Defaults to False.
        progress (gr.Progress, optional): Gradio progress tracker. Defaults to gr.Progress(track_tqdm=True).
    
    Returns:
        tuple: A tuple containing:
            - video_path (str): Path to the generated video file (.mp4)
            - current_seed (int): The seed used for generation (useful when randomize_seed=True)
    
    Raises:
        gr.Error: If input_image is None (no image uploaded).
    
    Note:
        - The function automatically resizes the input image to the target dimensions
        - Frame count is calculated as duration_seconds * FIXED_FPS (24)
        - Output dimensions are adjusted to be multiples of MOD_VALUE (32)
        - The function uses GPU acceleration via the @spaces.GPU decorator
        - Generation time varies based on steps and duration (see get_duration function)
    """
    if input_image is None:
        raise gr.Error("Please upload an input image.")
    
    num_frames = np.clip(int(round(duration_seconds * FIXED_FPS)), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    resized_image = resize_image(input_image)

    output_frames_list = pipe(
        image=resized_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=resized_image.height,
        width=resized_image.width,
        num_frames=num_frames,
        guidance_scale=float(guidance_scale),
        guidance_scale_2=float(guidance_scale_2),
        num_inference_steps=int(steps),
        generator=torch.Generator(device="cuda").manual_seed(current_seed),
    ).frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        video_path = tmpfile.name

    export_to_video(output_frames_list, video_path, fps=FIXED_FPS)

    return video_path, current_seed

with gr.Blocks() as demo:
    gr.Markdown("# Fast 4 steps Wan 2.1 I2V (14B) with CausVid LoRA")
    gr.Markdown("[CausVid](https://github.com/tianweiy/CausVid) is a distilled version of Wan 2.1 to run faster in just 4-8 steps, [extracted as LoRA by Kijai](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan21_CausVid_14B_T2V_lora_rank32.safetensors) and is compatible with 🧨 diffusers")
    with gr.Row():
        with gr.Column():
            input_image_component = gr.Image(type="pil", label="Input Image (auto-resized to target H/W)")
            prompt_input = gr.Textbox(label="Prompt", value=default_prompt_i2v)
            duration_seconds_input = gr.Slider(minimum=MIN_DURATION, maximum=MAX_DURATION, step=0.1, value=MAX_DURATION, label="Duration (seconds)", info=f"Clamped to model's {MIN_FRAMES_MODEL}-{MAX_FRAMES_MODEL} frames at {FIXED_FPS}fps.")
            
            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt_input = gr.Textbox(label="Negative Prompt", value=default_negative_prompt, lines=3)
                seed_input = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42, interactive=True)
                randomize_seed_checkbox = gr.Checkbox(label="Randomize seed", value=True, interactive=True)
                steps_slider = gr.Slider(minimum=1, maximum=30, step=1, value=6, label="Inference Steps") 
                guidance_scale_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale - high noise stage")
                guidance_scale_2_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=3, label="Guidance Scale 2 - low noise stage")

            generate_button = gr.Button("Generate Video", variant="primary")
        with gr.Column():
            video_output = gr.Video(label="Generated Video", autoplay=True, interactive=False)
    
    ui_inputs = [
        input_image_component, prompt_input,
        negative_prompt_input, duration_seconds_input,
        guidance_scale_input, guidance_scale_2_input, steps_slider, seed_input, randomize_seed_checkbox
    ]
    generate_button.click(fn=generate_video, inputs=ui_inputs, outputs=[video_output, seed_input])

    gr.Examples(
        examples=[ 
            [
                "wan_i2v_input.JPG",
                "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
            ],
        ],
        inputs=[input_image_component, prompt_input], outputs=[video_output, seed_input], fn=generate_video, cache_examples="lazy"
    )

if __name__ == "__main__":
    demo.queue().launch(mcp_server=True)
