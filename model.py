from unsloth import FastLanguageModel
import torch

def getBaseModel():
    model, _ = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        dtype=None,
        load_in_4bit=True
        )
    print(model.get_input_embeddings())
    return model

def getLoRAmodel(basemodel, r=4, targetModules=[]):
    lora_model = FastLanguageModel.get_peft_model(
                            basemodel,
                            r=r,
                            lora_alpha=2*r,
                            target_modules=targetModules,
                            bias="none",
                            use_gradient_checkpointing="unsloth")

    print("LORA model trainable params: ", lora_model.print_trainable_parameters())
    return lora_model
