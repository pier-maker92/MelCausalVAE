def count_parameters_by_module(model, title: str):
    table_width = 65
    print("=" * table_width)
    print(f"{title:^{table_width}}")
    print("-" * table_width)
    print(f"{'Module':<30} | {'Total Params':>15} | {'Trainable':>12}")
    print("-" * table_width)
    total_p = 0
    trainable_p = 0
    for name, module in model.named_children():
        t = sum(p.numel() for p in module.parameters())
        tr = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name:<30} | {t:15,d} | {tr:12,d}")
        total_p += t
        trainable_p += tr
    print("-" * table_width)
    print(f"{'GRAND TOTAL':<30} | {total_p:15,d} | {trainable_p:12,d}")
    print("-" * table_width + "\n")
