import torch
from transformers import BitsAndBytesConfig, AutoTokenizer

from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM
from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer

if __name__ == "__main__":
    patch_bnb_quantizer()

    model_dir = "../qwen3-moe-fused"
    model_quantized_dir = "../qwen3-moe-fused-quantized"

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")

    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
    )

    model = Qwen3MoeFusedForCausalLM.from_pretrained(model_dir, quantization_config=bnb_config)
    model.save_pretrained(model_quantized_dir)
    tokenizer.save_pretrained(model_quantized_dir)
