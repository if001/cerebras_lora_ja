import os

import torch
import transformers
import fire
from peft import PeftModel


def generate_prompt(instruction, input=None):
    if input:
        return f"""以下は、タスクを説明する命令と、さらなるコンテキストを提供する入力の組み合わせです。要求を適切に満たすような応答を書きなさい。

### 命令:
{instruction}

### 入力:
{input}

### 応答:"""
    else:
        return f"""以下は、ある作業を記述した指示です。要求を適切に満たすような応答を書きなさい。

### 命令:
{instruction}

### 応答:"""

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def infarence(
        model_name: str,
        lora_weights: str,
        prompt: str,
        gpu: bool = True,        
    ):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)    
    if device == 'cuda':        
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    else:        
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    prompt = generate_prompt(prompt)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if gpu:
        input_ids = input_ids.cuda()

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_p=0.75,
            top_k=85,
            temperature=1.99,
            typical_p=1,
            repetition_penalty=1.3,
            max_length=250,  # The maximum number of tokens to generate
            num_beams=5,    # The number of beams to use for beam search    
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)

def test():
    model_name = 'cerebras/Cerebras-GPT-2.7B'
    prompt = "Human:give me a 3 day travel plan for hawaii\n\nAssistant:"

    infarence(
        model_name,
        prompt
    ) 

if __name__ == '__main__':
    fire.Fire(infarence)
