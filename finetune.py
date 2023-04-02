import torch
import transformers
import datasets
import peft
from fire import Fire

cutoff_len = 512


def tokenize(tokenizer, item, add_eos_token=True):
    result = tokenizer(
        item,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )

    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def train(
        model_name: str,
        data_file: str,
        output_dir: str
        ):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0

    dataset = datasets.load_dataset('json', data_files=data_file)
    train_val = dataset["train"].train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_data = train_val["train"].shuffle().map(tokenize)
    val_data = train_val["test"].shuffle().map(tokenize)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        'cerebras/Cerebras-GPT-2.7B',    
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map={'': 0}
    )

    loraConfig = peft.LoraConfig(
        r=8,
        lora_alpha=16,
        # target_modules=["q_proj", "v_proj"],
        target_modules=["c_attn"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = peft.prepare_model_for_int8_training(model)
    model = peft.get_peft_model(model, loraConfig)

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=16, 
        gradient_accumulation_steps=8,  
        num_train_epochs=3,  
        learning_rate=1e-4, 
        fp16=True,
        optim="adamw_torch",
        logging_steps=10, 
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        output_dir=output_dir, 
        save_total_limit=3,
    )

    trainer = transformers.Trainer(
        model=model, 
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args, 
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False    
    trainer.train()
    model.save_pretrained(output_dir)


def test():
    model_name = 'cerebras/Cerebras-GPT-2.7B'
    data_file = 'alpaca_data_cleaned.json'
    output_dir = ''
    train(
        model_name,
        data_file
    )


if __name__ == '__main__':
    fire.Fire(train)