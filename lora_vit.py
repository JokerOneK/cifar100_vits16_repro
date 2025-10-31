import torch.nn as nn, loralib as lora
from train import build_model, make_loaders, train_one

def inject_lora_into_attn(module, r, alpha):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            new_layer = lora.Linear(child.in_features, child.out_features,
                                    r=r, lora_alpha=alpha, bias=(child.bias is not None))
            new_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                new_layer.bias.data = child.bias.data.clone()
            setattr(module, name, new_layer)
        else:
            inject_lora_into_attn(child, r, alpha)

def prepare_lora(model, r, alpha):
    for p in model.parameters():
        p.requires_grad = False
    # head trainable
    for p in model.get_classifier().parameters():
        p.requires_grad = True
    # LoRA in attention blocks (qkv + proj)
    for blk in model.blocks:
        inject_lora_into_attn(blk.attn, r=r, alpha=alpha)
    # mark lora params trainable
    for n,p in model.named_parameters():
        if 'lora_' in n or 'head' in n or 'fc_norm' in n:
            p.requires_grad = True
    return model

def run(r, alpha):
    cfg = dict(lr=2e-4, wd=0.0, epochs=10)
    model = prepare_lora(build_model(), r=r, alpha=alpha)
    train_dl, test_dl = make_loaders(32)
    res = train_one(model, train_dl, test_dl, cfg, seed=1, use_sam=False)
    print(f"LoRA r={r} a={alpha} ->", res)

if __name__ == "__main__":
    run(8,16)
    run(4,8)
