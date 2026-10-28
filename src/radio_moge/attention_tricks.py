"""Five prespecified changes at fixed 4x2048 width, paired over two seeds."""

from .capacity_screen import capacity_candidates


VARIANTS = {
    "baseline": {},
    "heads16": {"heads": 16},
    "heads64": {"heads": 64},
    "qknorm": {"qk_norm": True},
    "summary_readout": {"summary_readout": True},
}


def attention_candidates():
    rows = []
    for base in capacity_candidates(((4, 2048),)):
        for variant, changes in VARIANTS.items():
            rows.append(dict(base, **changes, variant=variant,
                             name=f"attention_{variant}_s{base['seed']}"))
    return rows
