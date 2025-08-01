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
from optimization_utils import ZeroGPUCompiledModel


P = ParamSpec('P')


TRANSFORMER_NUM_FRAMES_DIM = torch.export.Dim('num_frames', min=3, max=21)

TRANSFORMER_DYNAMIC_SHAPES = {
    'hidden_states': {
        2: TRANSFORMER_NUM_FRAMES_DIM,
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
        
        pipeline.load_lora_weights(
           "vrgamedevgirl84/Wan14BT2VFusioniX", 
           weight_name="FusionX_LoRa/Phantom_Wan_14B_FusionX_LoRA.safetensors", 
            adapter_name="phantom"
        )
        kwargs_lora = {}
        kwargs_lora["load_into_transformer_2"] = True
        pipeline.load_lora_weights(
           "vrgamedevgirl84/Wan14BT2VFusioniX", 
           weight_name="FusionX_LoRa/Phantom_Wan_14B_FusionX_LoRA.safetensors", 
            adapter_name="phantom_2", **kwargs_lora
        )
        pipeline.set_adapters(["phantom", "phantom_2"], adapter_weights=[1., 1.])
        pipeline.fuse_lora(adapter_names=["phantom"], lora_scale=3., components=["transformer"])
        pipeline.fuse_lora(adapter_names=["phantom_2"], lora_scale=1., components=["transformer_2"])
        pipeline.unload_lora_weights()
        
        with capture_component_call(pipeline, 'transformer') as call:
            pipeline(*args, **kwargs)
        
        dynamic_shapes = tree_map_only((torch.Tensor, bool), lambda t: None, call.kwargs)
        dynamic_shapes |= TRANSFORMER_DYNAMIC_SHAPES

        quantize_(pipeline.transformer, Float8DynamicActivationFloat8WeightConfig())
        quantize_(pipeline.transformer_2, Float8DynamicActivationFloat8WeightConfig())
        
        hidden_states: torch.Tensor = call.kwargs['hidden_states']
        hidden_states_transposed = hidden_states.transpose(-1, -2).contiguous()
        if hidden_states.shape[-1] > hidden_states.shape[-2]:
            hidden_states_landscape = hidden_states
            hidden_states_portrait = hidden_states_transposed
        else:
            hidden_states_landscape = hidden_states_transposed
            hidden_states_portrait = hidden_states

        exported_landscape_1 = torch.export.export(
            mod=pipeline.transformer,
            args=call.args,
            kwargs=call.kwargs | {'hidden_states': hidden_states_landscape},
            dynamic_shapes=dynamic_shapes,
        )
        
        exported_portrait_2 = torch.export.export(
            mod=pipeline.transformer_2,
            args=call.args,
            kwargs=call.kwargs | {'hidden_states': hidden_states_portrait},
            dynamic_shapes=dynamic_shapes,
        )

        compiled_landscape_1 = aoti_compile(exported_landscape_1, INDUCTOR_CONFIGS)
        compiled_portrait_2 = aoti_compile(exported_portrait_2, INDUCTOR_CONFIGS)

        compiled_landscape_2 = ZeroGPUCompiledModel(compiled_landscape_1.archive_file, compiled_portrait_2.weights)
        compiled_portrait_1 = ZeroGPUCompiledModel(compiled_portrait_2.archive_file, compiled_landscape_1.weights)

        return (
            compiled_landscape_1,
            compiled_landscape_2,
            compiled_portrait_1,
            compiled_portrait_2,
        )

    quantize_(pipeline.text_encoder, Int8WeightOnlyConfig())
    cl1, cl2, cp1, cp2 = compile_transformer()

    def combined_transformer_1(*args, **kwargs):
        hidden_states: torch.Tensor = kwargs['hidden_states']
        if hidden_states.shape[-1] > hidden_states.shape[-2]:
            return cl1(*args, **kwargs)
        else:
            return cp1(*args, **kwargs)

    def combined_transformer_2(*args, **kwargs):
        hidden_states: torch.Tensor = kwargs['hidden_states']
        if hidden_states.shape[-1] > hidden_states.shape[-2]:
            return cl2(*args, **kwargs)
        else:
            return cp2(*args, **kwargs)

    transformer_config = pipeline.transformer.config
    transformer_dtype = pipeline.transformer.dtype

    pipeline.transformer = combined_transformer_1
    pipeline.transformer.config = transformer_config # pyright: ignore[reportAttributeAccessIssue]
    pipeline.transformer.dtype = transformer_dtype # pyright: ignore[reportAttributeAccessIssue]

    pipeline.transformer_2 = combined_transformer_2
    pipeline.transformer_2.config = transformer_config # pyright: ignore[reportAttributeAccessIssue]
    pipeline.transformer_2.dtype = transformer_dtype # pyright: ignore[reportAttributeAccessIssue]
