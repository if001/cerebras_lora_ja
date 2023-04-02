# readme

```python
python finetune.py \
    --model_name='cerebras/Cerebras-GPT-2.7B'
    --data_file='./alpaca_data_cleaned.json'
    --output_dir=''
```

```python
python merged_lora_weights.py \
    --base_model='cerebras/Cerebras-GPT-2.7B'
    --model_name='cerebras/Cerebras-GPT-2.7B'
    --output_dir='./hf_ckpt'
```


```python
python infarence.py \
    --model_name='cerebras/Cerebras-GPT-2.7B'    
    --prompt=''
```

