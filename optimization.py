"""
"""

from typing import Any
from typing import Callable
from typing import ParamSpec

import spaces
import torch
from torch.utils._pytree import tree_map_only
from torchao.quantization import quantize_
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
from torchao.quantization import Int8WeightOnlyConfig

from optimization_utils import capture_component_call
from optimization_utils import aoti_compile
from optimization_utils import drain_module_parameters


P = ParamSpec('P')

LATENT_FRAMES_DIM = torch.export.Dim('num_latent_frames', min=8, max=81)

LATENT_PATCHED_HEIGHT_DIM = torch.export.Dim('latent_patched_height', min=30, max=52)
LATENT_PATCHED_WIDTH_DIM = torch.export.Dim('latent_patched_width', min=30, max=52)

TRANSFORMER_DYNAMIC_SHAPES = {
    'hidden_states': {
        2: LATENT_FRAMES_DIM,
        3: 2 * LATENT_PATCHED_HEIGHT_DIM,
        4: 2 * LATENT_PATCHED_WIDTH_DIM,
    },
}

INDUCTOR_CONFIGS = {
    'conv_1x1_as_mm': True,
    'epilogue_fusion': False,
    'coordinate_descent_tuning': True,
    'coordinate_descent_check_all_directions': True,
    'max_autotune': True,
    'triton.cudagraphs': True,
}


def optimize_pipeline_(pipeline: Callable[P, Any], *args: P.args, **kwargs: P.kwargs):

    @spaces.GPU(duration=1500)
    def compile_transformer():
        
        # This LoRA fusion part remains the same
        pipeline.load_lora_weights(
            "Kijai/WanVideo_comfy", 
            weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors", 
            adapter_name="lightx2v"
        )
        kwargs_lora = {}
        kwargs_lora["load_into_transformer_2"] = True
        pipeline.load_lora_weights(
            "Kijai/WanVideo_comfy", 
            weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors", 
            adapter_name="lightx2v_2", **kwargs_lora
        )
        pipeline.set_adapters(["lightx2v", "lightx2v_2"], adapter_weights=[1., 1.])
        pipeline.fuse_lora(adapter_names=["lightx2v"], lora_scale=3., components=["transformer"])
        pipeline.fuse_lora(adapter_names=["lightx2v_2"], lora_scale=1., components=["transformer_2"])
        pipeline.unload_lora_weights()
        
        with capture_component_call(pipeline, 'transformer') as call:
            pipeline(*args, **kwargs)
        
        dynamic_shapes = tree_map_only((torch.Tensor, bool), lambda t: None, call.kwargs)
        dynamic_shapes |= TRANSFORMER_DYNAMIC_SHAPES

        quantize_(pipeline.transformer, Float8DynamicActivationFloat8WeightConfig())
        quantize_(pipeline.transformer_2, Float8DynamicActivationFloat8WeightConfig())
        
        
        exported_1 = torch.export.export(
            mod=pipeline.transformer,
            args=call.args,
            kwargs=call.kwargs,
            dynamic_shapes=dynamic_shapes,
        )
        
        exported_2 = torch.export.export(
            mod=pipeline.transformer_2,
            args=call.args,
            kwargs=call.kwargs,
            dynamic_shapes=dynamic_shapes,
        )

        compiled_1 = aoti_compile(exported_1, INDUCTOR_CONFIGS)
        compiled_2 = aoti_compile(exported_2, INDUCTOR_CONFIGS)
        
        return compiled_1, compiled_2


    quantize_(pipeline.text_encoder, Int8WeightOnlyConfig())
    
    compiled_transformer_1, compiled_transformer_2 = compile_transformer()

    pipeline.transformer.forward = compiled_transformer_1
    drain_module_parameters(pipeline.transformer)

    pipeline.transformer_2.forward = compiled_transformer_2
    drain_module_parameters(pipeline.transformer_2)