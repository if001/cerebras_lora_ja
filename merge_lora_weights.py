import os

import torch
import transformers
from peft import PeftModel
import fire

def merge_weights(
        base_model: str,
        lora_weights: str,
        output_dir: str
    ):    
    device_map={"": "cpu"}
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    first_weight = model.transformer.h[0].attn.c_attn.weight
    first_weight_old = first_weight.clone()

    lora_model = PeftModel.from_pretrained(
        model,
        lora_weights,
        device_map=device_map,
        torch_dtype=torch.float16,
    )
    # lora_weight = lora_model.base_model.transformer.h[0].attn.c_attn.weight

    assert torch.allclose(first_weight_old, first_weight)

    # merge weights
    for layer in lora_model.base_model.transformer.h:
        layer.attn.c_attn.merge_weights = True

    lora_model.train(False)

    # did we do anything?
    assert not torch.allclose(first_weight_old, first_weight)

    lora_model_sd = lora_model.state_dict()
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model_sd.items()
        if "lora" not in k
    }

    base_model.save_pretrained(output_dir, state_dict=deloreanized_sd, max_shard_size="12GB")


def test():    
    base_model = 'cerebras/Cerebras-GPT-2.7B'
    model_name = ''
    output_dir = ''
    merge_weights(
        base_model,
        model_name,
        output_dir
    )

if __name__ == '__main__':
    fire.Fire(merge_weights)
