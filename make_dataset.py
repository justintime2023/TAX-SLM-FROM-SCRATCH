import random

SLABS = [
    (0, 400000, 0.00),
    (400000, 800000, 0.05),
    (800000, 1200000, 0.10),
    (1200000, 1600000, 0.15),
    (1600000, 2000000, 0.20),
    (2000000, 2400000, 0.25),
    (2400000, float("inf"), 0.30),
]

REBATE_LIMIT = 1200000
STD_DEDUCTION = 75000

def slab_tax(income):
    tax = 0.0
    for lo, hi, rate in SLABS:
        if income > lo:
            tax += (min(income, hi) - lo) * rate
    return int(round(tax))

def apply_rebate(taxable, slab_tax_amt):
    if taxable <= REBATE_LIMIT:
        return 0, "rebate"
    if taxable <= 1275000:
        return min(slab_tax_amt, taxable - REBATE_LIMIT), "marginal"
    return slab_tax_amt, "normal"

def make_example(income, salaried):
    std = STD_DEDUCTION if salaried else 0
    taxable = max(0, income - std)

    slab_amt = slab_tax(taxable)
    final_tax, mode = apply_rebate(taxable, slab_amt)

    lines = [
        "### TASK: NEW_TAX_REGIME",
        f"INCOME: {income}",
        f"SALARIED: {'YES' if salaried else 'NO'}",
        "### ANSWER:",
        f"Taxable income is Rs {taxable:,}.",
        f"Slab tax is about Rs {slab_amt:,}.",
    ]

    if mode == "rebate":
        lines.append("Since taxable income is up to Rs 12,00,000, rebate makes tax nil.")
    elif mode == "marginal":
        lines.append("Marginal relief applies near Rs 12,00,000.")
    else:
        lines.append("Rebate does not apply at this income level.")

    lines.append(f"Estimated tax payable is Rs {final_tax:,}.")
    lines.append("Note: Simplified estimate. Cess and surcharge not included.")

    return "\n".join(lines)

def main():
    random.seed(42)
    examples = []

    # Generate 12,000 diverse examples
    for _ in range(12000):
        income = random.randrange(300000, 3500000, 10000)
        salaried = random.random() < 0.65
        examples.append(make_example(income, salaried))

    text = "\n\n".join(examples) + "\n"
    
    import os
    os.makedirs("data/taxregime", exist_ok=True)
    
    with open("data/taxregime/input.txt", "w", encoding='utf-8') as f:
        f.write(text)

    print(f"✓ Dataset created with {len(examples):,} examples")
    print(f"✓ Total characters: {len(text):,}")

if __name__ == "__main__":
    main()
