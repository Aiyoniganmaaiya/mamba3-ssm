"""Utility functions for parameter counting and model summary."""


def count_parameters(model) -> tuple:
    """Count trainable and total parameters.

    Returns:
        (trainable_count, total_count)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def model_summary(model, name: str = "Model") -> str:
    """Produce a human-readable parameter summary string."""
    trainable, total = count_parameters(model)
    lines = [
        f"{name} parameter summary:",
        f"  Trainable: {trainable:>12,}",
        f"  Total:     {total:>12,}",
        f"  Size (fp32): {total * 4 / 1e9:.2f} GB",
    ]
    return "\n".join(lines)
