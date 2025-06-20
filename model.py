from transformers import AutoModelForSeq2SeqLM
import torch
from peft import LoraConfig, get_peft_model
from data.tokenization import T5Tokenizer

def getBaseModel():
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small",
                                                torch_dtype=torch.bfloat16)
    print(model.get_input_embeddings())
    tokenizer = T5Tokenizer()
    model.resize_token_embeddings(len(tokenizer.tokenizer))
    print("After adding new tokens: ", model.get_input_embeddings())
    return model

def viewModel(model):
    print(model)

def getLoRAModel(basemodel, r=2, targetModules=["q", "v"]):
    lora_config = LoraConfig(
        task_type="SEQ_2_SEQ_LM",
        r=2,
        target_modules=targetModules)

    LoRA_model = get_peft_model(basemodel, lora_config)
    print(LoRA_model.print_trainable_parameters())

    LoRA_model = LoRA_model.to(torch.bfloat16)
    # for name, param in LoRA_model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Parameter: {name}, Dtype: {param.dtype}, Requires Grad: {param.requires_grad}")
    return LoRA_model


if __name__ == "__main__":
    baseModel = getBaseModel()
    LoRAmodel = getLoRAModel(baseModel)
