from qwen3_moe_fused.convert import convert_model_to_fused
from transformers import AutoModelForCausalLM

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")
    model.save_pretrained("../local-qwen-moe-unfused")
    convert_model_to_fused("../local-qwen-moe-unfused", "../qwen3-moe-fused")
