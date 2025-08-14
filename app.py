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

# Dynamic sizing parameters
MOD_VALUE = 32
DEFAULT_H = 480
DEFAULT_W = 832
MAX_AREA = 480.0 * 832.0  # Maximum area for resolution calculation

# Slider ranges for manual adjustment
SLIDER_MIN_H, SLIDER_MAX_H = 128, 896
SLIDER_MIN_W, SLIDER_MAX_W = 128, 896

MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 81

MIN_DURATION = round(MIN_FRAMES_MODEL/FIXED_FPS, 1)
MAX_DURATION = round(MAX_FRAMES_MODEL/FIXED_FPS, 1)

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

for i in range(3): 
    gc.collect()
    torch.cuda.synchronize() 
    torch.cuda.empty_cache()

# Optimize with default dimensions for initial load
optimize_pipeline_(pipe,
    image=Image.new('RGB', (DEFAULT_W, DEFAULT_H)),
    prompt='prompt',
    height=DEFAULT_H,
    width=DEFAULT_W,
    num_frames=MAX_FRAMES_MODEL,
)

default_prompt_i2v = "make this image come alive, cinematic motion, smooth animation"
default_negative_prompt = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"


def calculate_optimal_dimensions(pil_image):
    """
    Calculate optimal dimensions for the output video based on input image aspect ratio.
    Maintains aspect ratio while fitting within the maximum area constraint.
    """
    if pil_image is None:
        return DEFAULT_H, DEFAULT_W
    
    orig_w, orig_h = pil_image.size
    if orig_w <= 0 or orig_h <= 0:
        return DEFAULT_H, DEFAULT_W
    
    # Calculate aspect ratio
    aspect_ratio = orig_h / orig_w
    
    # Calculate dimensions that maintain aspect ratio within max area
    calc_h = round(np.sqrt(MAX_AREA * aspect_ratio))
    calc_w = round(np.sqrt(MAX_AREA / aspect_ratio))
    
    # Ensure dimensions are multiples of MOD_VALUE
    calc_h = max(MOD_VALUE, (calc_h // MOD_VALUE) * MOD_VALUE)
    calc_w = max(MOD_VALUE, (calc_w // MOD_VALUE) * MOD_VALUE)
    
    # Clamp to slider ranges
    new_h = int(np.clip(calc_h, SLIDER_MIN_H, (SLIDER_MAX_H // MOD_VALUE) * MOD_VALUE))
    new_w = int(np.clip(calc_w, SLIDER_MIN_W, (SLIDER_MAX_W // MOD_VALUE) * MOD_VALUE))
    
    return new_h, new_w


def handle_image_upload(uploaded_image, current_h, current_w):
    """
    Update height and width sliders when an image is uploaded.
    """
    if uploaded_image is None:
        return gr.update(value=DEFAULT_H), gr.update(value=DEFAULT_W)
    
    try:
        new_h, new_w = calculate_optimal_dimensions(uploaded_image)
        return gr.update(value=new_h), gr.update(value=new_w)
    except Exception as e:
        gr.Warning("Error calculating dimensions, using defaults")
        return gr.update(value=DEFAULT_H), gr.update(value=DEFAULT_W)


def get_duration(
    input_image,
    prompt,
    steps,
    negative_prompt,
    duration_seconds,
    height,
    width,
    guidance_scale,
    guidance_scale_2,
    seed,
    randomize_seed,
    progress,
):
    return int(steps) * 15


@spaces.GPU(duration=get_duration)
def generate_video(
    input_image,
    prompt,
    steps=4,
    negative_prompt=default_negative_prompt,
    duration_seconds=MAX_DURATION,
    height=DEFAULT_H,
    width=DEFAULT_W,
    guidance_scale=1,
    guidance_scale_2=1,    
    seed=42,
    randomize_seed=False,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generate a video from an input image using the Wan 2.2 14B I2V model with Lightning LoRA.
    
    This function takes an input image and generates a video animation based on the provided
    prompt and parameters. It uses an FP8 quantized Wan 2.2 14B Image-to-Video model with Lightning LoRA
    for fast generation in 4-8 steps.
    
    Args:
        input_image (PIL.Image): The input image to animate. Will be resized to target dimensions.
        prompt (str): Text prompt describing the desired animation or motion.
        steps (int, optional): Number of inference steps. More steps = higher quality but slower.
            Defaults to 4. Range: 1-30.
        negative_prompt (str, optional): Negative prompt to avoid unwanted elements. 
            Defaults to default_negative_prompt (contains unwanted visual artifacts).
        duration_seconds (float, optional): Duration of the generated video in seconds.
            Defaults to MAX_DURATION. Clamped between MIN_DURATION and MAX_DURATION.
        height (int): Target height for the output video. Will be adjusted to multiple of MOD_VALUE (32).
        width (int): Target width for the output video. Will be adjusted to multiple of MOD_VALUE (32).
        guidance_scale (float, optional): Controls adherence to the prompt. Higher values = more adherence.
            Defaults to 1.0. Range: 0.0-10.0.
        guidance_scale_2 (float, optional): Controls adherence to the prompt in low noise stage.
            Defaults to 1.0. Range: 0.0-10.0.
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
    """
    if input_image is None:
        raise gr.Error("Please upload an input image.")
    
    # Ensure dimensions are multiples of MOD_VALUE
    target_h = max(MOD_VALUE, (int(height) // MOD_VALUE) * MOD_VALUE)
    target_w = max(MOD_VALUE, (int(width) // MOD_VALUE) * MOD_VALUE)
    
    num_frames = np.clip(int(round(duration_seconds * FIXED_FPS)), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    
    # Resize image to target dimensions
    resized_image = input_image.resize((target_w, target_h), Image.LANCZOS)

    output_frames_list = pipe(
        image=resized_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=target_h,
        width=target_w,
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
    gr.Markdown("# Fast 4 steps Wan 2.2 I2V (14B) with Lightning LoRA")
    gr.Markdown("Run Wan 2.2 in just 4-8 steps, with [Lightning LoRA](https://huggingface.co/Kijai/WanVideo_comfy/tree/main/Wan22-Lightning), fp8 quantization & AoT compilation - compatible with 🧨 diffusers and ZeroGPU⚡️")
    
    with gr.Row():
        with gr.Column():
            input_image_component = gr.Image(type="pil", label="Input Image (auto-resized to target H/W)")
            prompt_input = gr.Textbox(label="Prompt", value=default_prompt_i2v)
            duration_seconds_input = gr.Slider(
                minimum=MIN_DURATION, 
                maximum=MAX_DURATION, 
                step=0.1, 
                value=3.5, 
                label="Duration (seconds)", 
                info=f"Clamped to model's {MIN_FRAMES_MODEL}-{MAX_FRAMES_MODEL} frames at {FIXED_FPS}fps."
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt_input = gr.Textbox(label="Negative Prompt", value=default_negative_prompt, lines=3)
                seed_input = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42, interactive=True)
                randomize_seed_checkbox = gr.Checkbox(label="Randomize seed", value=True, interactive=True)
                
                with gr.Row():
                    height_input = gr.Slider(
                        minimum=SLIDER_MIN_H, 
                        maximum=SLIDER_MAX_H, 
                        step=MOD_VALUE, 
                        value=DEFAULT_H, 
                        label=f"Output Height (multiple of {MOD_VALUE})"
                    )
                    width_input = gr.Slider(
                        minimum=SLIDER_MIN_W, 
                        maximum=SLIDER_MAX_W, 
                        step=MOD_VALUE, 
                        value=DEFAULT_W, 
                        label=f"Output Width (multiple of {MOD_VALUE})"
                    )
                
                steps_slider = gr.Slider(minimum=1, maximum=30, step=1, value=6, label="Inference Steps") 
                guidance_scale_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale - high noise stage")
                guidance_scale_2_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale 2 - low noise stage")

            generate_button = gr.Button("Generate Video", variant="primary")
            
        with gr.Column():
            video_output = gr.Video(label="Generated Video", autoplay=True, interactive=False)
    
    # Auto-update dimensions when image is uploaded
    input_image_component.upload(
        fn=handle_image_upload,
        inputs=[input_image_component, height_input, width_input],
        outputs=[height_input, width_input]
    )
    
    # Reset dimensions when image is cleared
    input_image_component.clear(
        fn=handle_image_upload,
        inputs=[input_image_component, height_input, width_input],
        outputs=[height_input, width_input]
    )
    
    ui_inputs = [
        input_image_component, prompt_input, steps_slider,
        negative_prompt_input, duration_seconds_input,
        height_input, width_input,
        guidance_scale_input, guidance_scale_2_input, 
        seed_input, randomize_seed_checkbox
    ]
    
    generate_button.click(fn=generate_video, inputs=ui_inputs, outputs=[video_output, seed_input])

    gr.Examples(
        examples=[ 
            [
                "wan_i2v_input.JPG",
                "POV selfie video, white cat with sunglasses standing on surfboard, relaxed smile, tropical beach behind (clear water, green hills, blue sky with clouds). Surfboard tips, cat falls into ocean, camera plunges underwater with bubbles and sunlight beams. Brief underwater view of cat's face, then cat resurfaces, still filming selfie, playful summer vacation mood.",
                4,
            ],
            [
                "wan22_input_2.jpg",
                "A sleek lunar vehicle glides into view from left to right, kicking up moon dust as astronauts in white spacesuits hop aboard with characteristic lunar bouncing movements. In the distant background, a VTOL craft descends straight down and lands silently on the surface. Throughout the entire scene, ethereal aurora borealis ribbons dance across the star-filled sky, casting shimmering curtains of green, blue, and purple light that bathe the lunar landscape in an otherworldly, magical glow.",
                4,
            ],
            [
                "kill_bill.jpeg",
                "Uma Thurman's character, Beatrix Kiddo, holds her razor-sharp katana blade steady in the cinematic lighting. Suddenly, the polished steel begins to soften and distort, like heated metal starting to lose its structural integrity. The blade's perfect edge slowly warps and droops, molten steel beginning to flow downward in silvery rivulets while maintaining its metallic sheen. The transformation starts subtly at first - a slight bend in the blade - then accelerates as the metal becomes increasingly fluid. The camera holds steady on her face as her piercing eyes gradually narrow, not with lethal focus, but with confusion and growing alarm as she watches her weapon dissolve before her eyes. Her breathing quickens slightly as she witnesses this impossible transformation. The melting intensifies, the katana's perfect form becoming increasingly abstract, dripping like liquid mercury from her grip. Molten droplets fall to the ground with soft metallic impacts. Her expression shifts from calm readiness to bewilderment and concern as her legendary instrument of vengeance literally liquefies in her hands, leaving her defenseless and disoriented.",
                6,
            ],
        ],
        inputs=[input_image_component, prompt_input, steps_slider], 
        outputs=[video_output, seed_input], 
        fn=generate_video, 
        cache_examples="lazy"
    )

if __name__ == "__main__":
    demo.queue().launch(mcp_server=True)