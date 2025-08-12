import os

from unsloth import FastModel

# Import unsloth before others
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import Trainer, TrainingArguments
from qwen3_moe_fused.fast_lora import patch_Qwen3MoeFusedSparseMoeBlock_forward
from qwen3_moe_fused.lora import patch_lora_config
from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM
from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer
from argparse import ArgumentParser
from functools import partial
from transformers import DataCollatorForSeq2Seq
import torch._dynamo
torch._dynamo.config.cache_size_limit = 512

# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

def tokenize_multiturn_chat(messages, tokenizer, max_length: int) -> dict:
    """
    Tokenize multi-turn chat data for language model training.

    Parameters
    ----------
    messages : list or dict
        Chat messages to tokenize. Can be a list of message dictionaries
        or a nested list of chats.
    tokenizer : PreTrainedTokenizer
        Tokenizer to use for processing text.
    max_length : int
        Maximum sequence length for tokenization.

    Returns
    -------
    dict
        Dictionary containing:
        - input_ids : list
            Tokenized input sequences
        - attention_mask : list
            Attention mask for the sequences
        - labels : list
            Labels for training, with -100 for non-assistant tokens

    Notes
    -----
    - Handles both single conversations and lists of conversations
    - Sets labels for assistant responses and special tokens
    - Supports different model architectures (Gemma, Llama, etc.)
    """
    if isinstance(messages[0], dict):
        chat_formatted = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    else:
        chat_formatted = ""
        for subchat in messages:
            try:
                chat_formatted += tokenizer.apply_chat_template(subchat, tokenize=False, enable_thinking=False)
            except:
                print(f"Error on chat: {subchat}")
    chat_tokenized = tokenizer(
        chat_formatted,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    chat_tokenized["labels"] = [-100] * len(chat_tokenized["input_ids"])
    start_of_turn_token = tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_start_token = tokenizer.convert_tokens_to_ids("assistant")
    assistant_end_token = tokenizer.convert_tokens_to_ids("<|im_end|>")
    think_start_token = tokenizer.convert_tokens_to_ids("<think>")
    think_end_token = tokenizer.convert_tokens_to_ids("</think>")
    assistant_starts_idxs = [i for i, token_id in enumerate(chat_tokenized["input_ids"]) if token_id == assistant_start_token and chat_tokenized["input_ids"][i-1] == start_of_turn_token]
    all_end_idxs = [i for i, token_id in enumerate(chat_tokenized["input_ids"]) if token_id == assistant_end_token]
    assistant_end_idxs = []
    for start_idx in assistant_starts_idxs:
        try:
            next_ends = [idx for idx in all_end_idxs if idx > start_idx][0]
        except:
            next_ends = -2
        assistant_end_idxs.append(next_ends)
    if len(assistant_starts_idxs) != len(assistant_end_idxs):
        print(f"*** No coinciden: \n{assistant_starts_idxs}\n{assistant_end_idxs}")
    pairs = [(start, end) for start, end in zip(assistant_starts_idxs, assistant_end_idxs)]
    for start, end in pairs:
        chat_tokenized["labels"][start+2:end + 1] = chat_tokenized["input_ids"][start+2:end + 1]
    
    think_starts_idxs = [i for i, token_id in enumerate(chat_tokenized["input_ids"]) if token_id == think_start_token]
    if len(think_starts_idxs) > 0:  
        all_think_end_idxs = [i for i, token_id in enumerate(chat_tokenized["input_ids"]) if token_id == think_end_token]
        think_end_idxs = []
        for start_idx in think_starts_idxs:
            try:
                next_end = [idx for idx in all_think_end_idxs if idx > start_idx][0]
            except:
                next_end = -2
            think_end_idxs.append(next_end)
        pairs_think = [(start, end) for start, end in zip(think_starts_idxs, think_end_idxs)]
        for start, end in pairs_think:
            chat_tokenized["labels"][start: end + 1] = [-100] * (end - start + 1)
    assert len(chat_tokenized["labels"]) == len(chat_tokenized["input_ids"])
    return chat_tokenized


def tokenize_chat(samples, tokenizer, chat_field: str = "chat", max_length: int = 8192) -> dict:
        """
        Tokenize chat samples for training.

        Parameters
        ----------
        samples : dict
            Samples containing chat messages.
        tokenizer : PreTrainedTokenizer
            Tokenizer to use.
        add_eos_token : bool, optional
            Whether to add EOS tokens.
        dataset_config : Any, optional
            Configuration for dataset processing.

        Returns
        -------
        dict
            Tokenized samples ready for training.

        Notes
        -----
        Handles batch processing of chat samples with proper
        tokenization and label creation.
        """
        new_samples = {"input_ids": [], "attention_mask": [], "labels": []}
        pasado_longitud = 0
        pasado_raro = 0
        for i in range(len(samples[chat_field])):
            messages = samples[chat_field][i]
            if isinstance(messages, list) and (isinstance(messages[0], dict) or isinstance(messages[0], list)):
                chat_tokenized = tokenize_multiturn_chat(messages, tokenizer, max_length)
                if len(chat_tokenized["input_ids"]) <= max_length:
                    for k, v in chat_tokenized.items():
                        if k in new_samples:
                            new_samples[k].append(v)
                else:
                    pasado_longitud += 1
                    print(f"Nos hemos pasado de longitud: {pasado_longitud} ## Longitud: {len(chat_tokenized['input_ids'])}")
            else:
                pasado_raro += 1
                print(f"Nos hemos pasado por algo raro: {pasado_raro}")
                
        return new_samples

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_alias", type=str, required=True)
    parser.add_argument("--model_alias", type=str, required=True)
    parser.add_argument("--chat_field", type=str, default="chat")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=10)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    args = parser.parse_args()

    patch_bnb_quantizer()
    # We can set a smaller rank for MoE layers
    # With rslora, we don't need to set a different alpha for them
    # TODO: Support rank_pattern in Unsloth
    patch_lora_config(
        rank_pattern={
            "q_proj": args.lora_r,
            "k_proj": args.lora_r,
            "v_proj": args.lora_r,
            "o_proj": args.lora_r,
            # "gate": 16,  # It's possible to create a LoRA on the routing gate, but this is unstable
            "gate_proj": int(args.lora_r/4),
            "up_proj": int(args.lora_r/4),
            "down_proj": int(args.lora_r/4),
        }
    )
    patch_Qwen3MoeFusedSparseMoeBlock_forward()


    model, tokenizer = FastModel.from_pretrained(args.model_id, auto_model=Qwen3MoeFusedForCausalLM)

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

    dataset = dataset.map(partial(tokenize_chat, tokenizer=tokenizer, chat_field=args.chat_field, max_length=args.max_length), batched=True, num_proc=8)

    fixed_train_args = {
        "output_dir": f"../models/{args.model_alias}",
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": int(args.batch_size/2),
        "gradient_accumulation_steps": int(16/args.batch_size),
        "warmup_ratio": args.warmup_ratio,
        "learning_rate": args.learning_rate,
        "bf16": True,
        "logging_steps": 50,
        "lr_scheduler_type": args.lr_scheduler_type,
        "weight_decay": 0.001,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "num_train_epochs": 1,
        "logging_first_step": True,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "max_grad_norm": 0.3,
        "optim": "paged_adamw_32bit",
        "gradient_checkpointing": True,
        "group_by_length": False,
        "save_total_limit": 50,
        "adam_beta2": 0.999,
        "dataloader_num_workers": 24,
        "torch_compile": True,
        "torch_compile_mode": "max-autotune",
    }

    training_args = TrainingArguments(
        **fixed_train_args
    )

    trainer = Trainer(
        model=model,
        data_collator=DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True),
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    trainer_stats = trainer.train()
    print(f"Trainer stats: {trainer_stats}")
    try:
        trainer.model.push_to_hub_merged(
            args.model_alias,
            trainer.tokenizer,
            save_method = "merged_16bit",
            private=True
        )
    except Exception as e:
        print(f"Error al subir el modelo a hub: {e}")
        try:
            trainer.model.push_to_hub_merged(
                f"avacaondata/{args.model_alias}",
                trainer.tokenizer,
                save_method = "merged_16bit",
                private=True
            )
        except Exception as e:
            trainer.model.save_pretrained(f"../models/{args.model_alias}-final")
