import os

from unsloth import FastModel, unsloth_train
from unsloth import PatchDPOTrainer
PatchDPOTrainer()
import json
# Import unsloth before others
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer 
from transformers import Trainer, TrainingArguments
from qwen3_moe_fused.fast_lora import patch_Qwen3MoeFusedSparseMoeBlock_forward
from qwen3_moe_fused.lora import patch_lora_config
from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM
from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer
from argparse import ArgumentParser
from functools import partial
from transformers import AutoTokenizer
import torch._dynamo
torch._dynamo.config.cache_size_limit = 512


def process(sample: dict) -> dict:
    """
    Procesa una muestra para entrenamiento DPO.

    Parameters
    ----------
    sample : dict
        Diccionario con datos de la muestra que incluye 'prompt',
        'chosen' y 'rejected'.

    Returns
    -------
    dict
        Muestra procesada con:
        - prompt: texto tokenizado y formateado
        - chosen: respuesta preferida formateada
        - rejected: respuesta rechazada formateada

    Notes
    -----
    - Aplica plantillas de chat específicas del modelo
    - Maneja tanto respuestas individuales como listas
    - Trunca secuencias según longitudes máximas configuradas
    """
    # sample = remove_input_field(sample, messages_field="prompt") # TODO: Quitar para modelos que si puedan usar eso.
    sample["prompt"] = tokenizer.decode(tokenizer.apply_chat_template(sample["prompt"], tokenize=True, max_length=int(args.max_length - args.max_target_length), truncation=True, enable_thinking=False))
    if isinstance(sample["chosen"], list):
        chosen = sample["chosen"]
        rejected = sample["rejected"]
    elif isinstance(sample["chosen"], dict):
        chosen = [sample["chosen"]]
        rejected = [sample["rejected"]]
    sample["chosen"] = tokenizer.apply_chat_template(chosen, tokenize=False, enable_thinking=False)
    # sample["chosen"] = sample["chosen"].replace("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", "")
    sample["rejected"] = tokenizer.apply_chat_template(rejected, tokenize=False, enable_thinking=False)
    # sample["rejected"] = sample["rejected"].replace("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", "")
    return sample


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default="avacaondata/superlenia-psicochat-1909")# "avacaondata/qwen2-7b-chat-es-v2")
    parser.add_argument("--max_length", default=32768, type=float)
    parser.add_argument("--max_target_length", default=3250, type=int)
    parser.add_argument("--save_name",default="superlenia-psicochat-1909-sppo")
    parser.add_argument("--dataset_name", default="avacaondata/total-dpo-250325")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--bos_token_id", default=151644, type=int)  # Para Qwen2.5: 151644 # 199999 # Para Llama: 128000  # para Phi-4-Mini: 199999 # Para gemma3: 2
    parser.add_argument("--pad_token_id", default=151643, type=int)  # Para Qwen2.5: 151643 #  Para Phi-4-Mini: 199999 # Para Llama: 128004 # Para Qwen2.5: 0
    # Añadir argumentos para los parámetros de LoRA
    parser.add_argument("--lora_r", default=128, type=int, help="Rango de LoRA")
    parser.add_argument("--lora_alpha", default=128, type=int, help="Alpha de LoRA")
    parser.add_argument("--lora_dropout", default=0.1, type=float, help="Dropout de LoRA")
    # Añadir argumento para learning rate
    parser.add_argument("--learning_rate", default=5.0e-6, type=float, help="Learning rate para el entrenamiento")
    # Añadir argumento para beta
    parser.add_argument("--beta", default=0.01, type=float, help="Parámetro beta para DPO")
    # Añadir argumento para eval_steps
    parser.add_argument("--eval_steps", default=500, type=int, help="Número de pasos entre evaluaciones")
    args = parser.parse_args()


    patch_bnb_quantizer()
    patch_lora_config(
        rank_pattern={
            "q_proj": args.lora_r,
            "k_proj": args.lora_r,
            "v_proj": args.lora_r,
            "o_proj": args.lora_r,
            "gate_proj": int(args.lora_r/4),
            "up_proj": int(args.lora_r/4),
            "down_proj": int(args.lora_r/4),
        }
    )
    patch_Qwen3MoeFusedSparseMoeBlock_forward()


    model, tokenizer = FastModel.from_pretrained(args.model, auto_model=Qwen3MoeFusedForCausalLM)
    model = FastModel.get_peft_model(
        model,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # "gate",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_rslora=True,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = load_dataset(args.dataset_name)
    dataset["train"] = dataset["train"].shuffle(seed=42)
    print(dataset)
    dataset = dataset.filter(lambda s: isinstance(s["chosen"], list) and all([isinstance(message["content"], str) for message in s["chosen"]]))
    print(dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    # tokenizer.add_special_tokens({"bos_token": "<bos>"}) # <|endoftext|> # Para Llama: <|begin_of_text|>
    tokenizer.bos_token_id = args.bos_token_id  # Para Qwen2.5: 151644 # 199999 # Para Llama: 128000  # para Phi-4-Mini: 199999 # Para gemma3: 2
    tokenizer.pad_token_id = args.pad_token_id  # Para Qwen2.5: 151643 #  Para Phi-4-Mini: 199999 # Para Llama: 128004 # Para Qwen2.5: 0
    # tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"}) # <|endoftext|> # Para Llama: <|finetune_right_pad_id|>

    dataset = dataset.map(process, batched=False, num_proc=24)

    # print(dataset)
    # print(dataset["train"][0])
    
    fixed_train_args = {
        "output_dir": f"./{args.save_name.split('/')[-1]}",
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": 0.1,
        "learning_rate": args.learning_rate,
        "bf16": True,
        "logging_steps": 10,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.001,
        "eval_steps": args.eval_steps,
        "save_steps": args.eval_steps,
        "num_train_epochs": 1,
        "logging_first_step": True,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "max_grad_norm": 0.3,
        "optim": "paged_adamw_32bit",
        "gradient_checkpointing": True,
        "group_by_length": False,
        "save_total_limit": 10,
        "adam_beta2": 0.999,
        "dataloader_num_workers": 24, # 24
        "beta": args.beta,
        "max_length": int(args.max_length),
        "max_prompt_length": int(args.max_length - args.max_target_length),
        "max_completion_length": int(args.max_target_length),
        "loss_type":"sppo_hard",
        "report_to": "none",
        "torch_compile": True,
        "torch_compile_mode": "max-autotune",
    }

    dpo_config = DPOConfig(**fixed_train_args)

    trainer = DPOTrainer(
        model,
        None,
        args=dpo_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        # peft_config=lora_config,
    )

    unsloth_train(trainer)
    # trainer.train()
    trainer.save_model(dpo_config.output_dir)
    metrics = trainer.evaluate()
    
    # Guardar métricas con el nombre del modelo
    metrics_filename = f"metrics_{args.save_name.split('/')[-1]}.json"
    with open(metrics_filename, "w") as f:
        json.dump(metrics, f)

