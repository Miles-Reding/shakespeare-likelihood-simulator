# shakespeare-likelihood-simulator

Uses statistical modeling to simulate the likelihood of a Shakespeare-like figure emerging in a given era.

## What this models

Not *the* Shakespeare, but "a Shakespeare" — operationally defined as a person who clears two kinds of conditions simultaneously:

1. **Intrinsic talent** — some top-percentile combination of linguistic ability, dramatic intuition, and empathy.
2. **Structural conditions** — the factors that let talent become recognized, preserved, and influential: a commercial theater industry, robust patronage networks, printing infrastructure, sufficient audience literacy, political stability, and a maturing English literary tradition. On top of that, the person needs to survive childhood, be male (given 16th-century constraints), come from the right class background (craftsman/merchant — educated but not noble), and be geographically positioned near London.

The key design choice: these are **AND conditions, not OR**. A figure with everything except patronage access probably produces work that isn't recognized or preserved. A figure born without theater infrastructure is a playwright without a theater. All the gates have to clear at once, which is what makes such a figure genuinely rare — and why the Elizabethan era stands out.

## Approach: analytical, not agent-based

Shakespeare-level talent is rare enough (~1 in 100,000 at minimum) that a naive agent-based simulation — spawning individuals and checking who clears every gate — would almost never produce a hit in any single run at a tractable population size.

Instead, `P(Shakespeare)` is computed **analytically per person-year** and multiplied by population to get an expected count. Monte Carlo is used only at the era-parameter level, adding Gaussian noise to structural conditions to simulate year-to-year uncertainty within an era and produce confidence bands. This also makes bottleneck analysis straightforward: `P(Shakespeare)` decomposes into multiplicative factors, so you can see directly which gate is binding in each era.

```
P(Shakespeare) = P(male)
               × P(survive childhood)
               × P(right class)
               × P(literate | class)
               × P(urban-adjacent | class)
               × P(talent in top X%)
               × Π gate_multipliers(structural conditions)
```

Structural gates are **soft**, not hard cutoffs: a below-threshold condition doesn't make emergence impossible, just far less likely (down to a small penalty fraction of baseline), which is closer to how these constraints actually behaved historically.

## Running it

```bash
pip install -r requirements.txt
python shakespeare_simulation.py
```

This prints a per-era summary (expected figures, gate multipliers, ratio vs. the Elizabethan era) and produces a 5-panel figure (`shakespeare_simulation.png`):

1. Expected annual output over time, with confidence bands, across all eras (1400–1800)
2. Total expected figures per era
3. Structural conditions — Late Medieval vs. Elizabethan, against required thresholds
4. A structural bottleneck index (combined gate multiplier) per era
5. Sensitivity of era rankings to the talent-threshold assumption

## What it finds

The Elizabethan era (1576–1616) comes out roughly 50–200× more likely than the Late Medieval period to produce a Shakespeare-level figure — not because individuals were more talented, but because the structural gates (theater, patronage, printing, literacy) were all simultaneously cleared for essentially the first time in English history. In the Late Medieval period, theater infrastructure and printing access are the binding constraints (both near zero); in the Interregnum, it's theater infrastructure again (theaters were closed 1642–1660).

Running the sensitivity panel across talent thresholds from the top 0.5% down to the top 0.02% shows the **era rankings are stable** even as absolute expected counts shift by orders of magnitude — the core finding (the Elizabethan window was uniquely favorable) doesn't depend on getting the hardest-to-justify parameter exactly right.

The model also abstracts away a "uniqueness" component: Shakespeare wasn't just a great playwright working under favorable conditions, he was the *first* great playwright working in that specific vernacular-theater-plus-printing-plus-patronage nexus. That first-mover effect isn't something a per-person-year probability model captures.

## Key adjustments worth making

A few natural next experiments from here, none implemented yet:

1. **Cultural saturation** — add a function where, once a Shakespeare-level figure has emerged, the probability resets lower for a generation, modeling the idea that a canonical figure in a genre suppresses successors (no one writes *King Lear* again right after *King Lear*). This would keep the 18th-century expected-count numbers more realistic — as written, the model treats each era's conditions as producing output independently, which overstates how many "Shakespeares" a sustained favorable environment should yield.
2. **Counterfactuals** — what if the theaters had never closed under the Interregnum? What if plague had killed Shakespeare in 1592 (a real near-miss)? The `Era` and `ShakespeareProfile` dataclasses make this trivial: modify one parameter and re-run.
3. **Cross-national comparison** — add Italian, French, or Spanish eras using the same parameter schema. Italy had earlier theater infrastructure and printing but lower political stability — does the model predict a Dante- or Machiavelli-equivalent emerging before an English one? That would be a real test of the model's validity.
4. **Tighter talent threshold** — dropping `talent_top_fraction` to `0.0001` or lower would put absolute expected counts in the "1–3 over 400 years" range that feels historically right, without changing the era-to-era comparisons.

`Era` is just a dataclass, so any alternative historical scenario (Elizabethan England without plague, printing arriving 50 years earlier, etc.) can be defined and dropped into `ERAS` without touching the probability logic.

## Sources

Parameter values are grounded in (though not literally sourced line-by-line from) the following scholarship:

- David Cressy, *Literacy and the Social Order* (literacy)
- Andrew Gurr, *The Shakespearean Stage* (theater history)
- E. A. Wrigley & R. S. Schofield (population)
- Gary Taylor, *Reinventing Shakespeare* (patronage)

## Caveats

This is a toy model, not a rigorous historical claim. It treats structural factors as conditionally independent given era parameters (in reality they're correlated), collapses "talent" into a single scalar percentile, and its absolute probability estimates shouldn't be taken literally — the interesting output is the *relative* comparison across eras and the bottleneck decomposition, not the point estimates.
