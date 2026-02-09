if __name__ == "__main__":
    from train import build_model, make_loaders, train_one
    
    cfg = dict(lr=5e-4, wd=0.05, epochs=10, scheduler='cosine', total_steps_override=15640)
    model = build_model()
    train_dl, test_dl = make_loaders(bs=32)
    res = train_one(model, train_dl, test_dl, cfg, seed=1, use_sam=False)
    print(res)