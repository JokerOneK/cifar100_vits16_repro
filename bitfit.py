def prepare_bitfit(model):
    # freeze all
    for p in model.parameters():
        p.requires_grad = False
    # head trainable
    for p in model.get_classifier().parameters():
        p.requires_grad = True
    # all bias trainable
    for name, p in model.named_parameters():
        if name.endswith('.bias'):
            p.requires_grad = True
    return model

if __name__ == "__main__":
    from train import build_model, make_loaders, train_one
    

    
    cfg = dict(lr=2e-4, wd=0.0, epochs=10)
    model = prepare_bitfit(build_model())
    train_dl, test_dl = make_loaders(32)
    res = train_one(model, train_dl, test_dl, cfg, seed=1, use_sam=False)
    print(res)