"""
shakespeare_simulation.py
─────────────────────────────────────────────────────────────────
A Monte Carlo simulation modeling how likely a Shakespeare-level
creative figure is to emerge under different historical conditions.

Design philosophy:
  - Each era is defined by structural parameters (theater, literacy, etc.)
  - For each year, we analytically compute P(Shakespeare) per person
  - Monte Carlo noise captures year-to-year uncertainty within eras
  - We aggregate to get expected figures per era with confidence bands
  - A bottleneck analysis decomposes which structural gates matter most
  - A sensitivity panel tests robustness across talent-threshold assumptions

Dependencies: numpy, matplotlib
Run: python shakespeare_simulation.py
─────────────────────────────────────────────────────────────────
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Dict, List

YEAR_START, YEAR_END = 1400, 1800


# ══════════════════════════════════════════════════════════════════
#  1. ERA PARAMETERS
#     Each era encodes the structural conditions of a historical
#     period.  All continuous fields are on a 0–1 normalized scale.
#
#     Values are grounded in scholarship:
#       - David Cressy, "Literacy and the Social Order" (literacy)
#       - Andrew Gurr, "The Shakespearean Stage" (theater history)
#       - E. A. Wrigley & R. S. Schofield (population)
#       - Gary Taylor, "Reinventing Shakespeare" (patronage)
# ══════════════════════════════════════════════════════════════════

@dataclass
class Era:
    """
    Structural conditions for a historical period.

    The core thesis: Shakespeare required not just extraordinary individual
    talent, but a specific confluence of structural conditions.  Change any
    one of them and the figure probably doesn't emerge in recognizable form.
    """
    name: str
    year_start: int
    year_end: int

    # ── Demographic ───────────────────────────────────────────────
    population: int              # approximate England population
    childhood_survival: float    # P(survive to working age, ~15)
    plague_years: List[int]      # years with elevated mortality

    # ── Individual-access probabilities ───────────────────────────
    male_literacy_rate: float    # fraction of males who can read
    urban_fraction: float        # fraction near a cultural center (e.g. London)

    # ── Structural conditions (0–1 scale) ─────────────────────────
    # These are era-level "availability" scores.  They gate whether
    # talent can be expressed, recognized, and preserved at all.
    theater_infrastructure: float  # 0 = none, 1 = robust commercial theater
    patronage_robustness:   float  # how available is arts patronage?
    printing_access:        float  # can work be disseminated and preserved?
    political_stability:    float  # does the climate allow creative work?
    vernacular_culture:     float  # maturity of an English literary tradition


ERAS: List[Era] = [
    Era(
        name="Late Medieval",
        year_start=1400, year_end=1476,
        population=2_500_000,
        childhood_survival=0.50,
        plague_years=list(range(1400, 1476, 14)),
        male_literacy_rate=0.08,
        urban_fraction=0.05,
        theater_infrastructure=0.05,    # mystery/morality plays; no commercial theater
        patronage_robustness=0.20,
        printing_access=0.00,           # Gutenberg's press not yet in England
        political_stability=0.40,       # Wars of the Roses
        vernacular_culture=0.12,
    ),
    Era(
        name="Early Tudor",
        year_start=1476, year_end=1540,
        population=2_800_000,
        childhood_survival=0.52,
        plague_years=[1499, 1513, 1528],
        male_literacy_rate=0.14,
        urban_fraction=0.06,
        theater_infrastructure=0.10,    # interludes, court masques; no fixed theaters
        patronage_robustness=0.32,
        printing_access=0.22,           # Caxton's press arrived 1476
        political_stability=0.60,
        vernacular_culture=0.25,
    ),
    Era(
        name="Mid Tudor",
        year_start=1540, year_end=1576,
        population=3_100_000,
        childhood_survival=0.54,
        plague_years=[1543, 1563],
        male_literacy_rate=0.20,
        urban_fraction=0.07,
        theater_infrastructure=0.18,    # professional troupes; still no fixed theaters
        patronage_robustness=0.38,
        printing_access=0.40,
        political_stability=0.48,       # Edward VI → Mary I → early Elizabeth
        vernacular_culture=0.40,
    ),
    Era(
        name="Elizabethan",             # ← Shakespeare's actual era
        year_start=1576, year_end=1616,
        population=4_100_000,
        childhood_survival=0.57,
        plague_years=[1592, 1593, 1603, 1608, 1609],
        male_literacy_rate=0.30,
        urban_fraction=0.09,
        theater_infrastructure=0.90,    # The Theatre (1576), Globe (1599)
        patronage_robustness=0.72,
        printing_access=0.65,
        political_stability=0.80,
        vernacular_culture=0.75,
    ),
    Era(
        name="Jacobean / Caroline",
        year_start=1616, year_end=1642,
        population=4_800_000,
        childhood_survival=0.58,
        plague_years=[1625, 1636],
        male_literacy_rate=0.35,
        urban_fraction=0.10,
        theater_infrastructure=0.78,    # still robust but under growing pressure
        patronage_robustness=0.62,
        printing_access=0.70,
        political_stability=0.52,
        vernacular_culture=0.80,
    ),
    Era(
        name="Interregnum / Restoration",
        year_start=1642, year_end=1700,
        population=5_200_000,
        childhood_survival=0.57,
        plague_years=[1665],
        male_literacy_rate=0.40,
        urban_fraction=0.12,
        theater_infrastructure=0.38,    # theaters closed 1642–1660; reopened, altered
        patronage_robustness=0.48,
        printing_access=0.75,
        political_stability=0.42,
        vernacular_culture=0.75,
    ),
    Era(
        name="Early 18th Century",
        year_start=1700, year_end=1760,
        population=6_000_000,
        childhood_survival=0.58,
        plague_years=[],
        male_literacy_rate=0.50,
        urban_fraction=0.16,
        theater_infrastructure=0.60,
        patronage_robustness=0.52,
        printing_access=0.85,
        political_stability=0.75,
        vernacular_culture=0.85,
    ),
    Era(
        name="Late 18th Century",
        year_start=1760, year_end=1800,
        population=7_500_000,
        childhood_survival=0.60,
        plague_years=[],
        male_literacy_rate=0.58,
        urban_fraction=0.22,
        theater_infrastructure=0.62,
        patronage_robustness=0.48,
        printing_access=0.90,
        political_stability=0.70,
        vernacular_culture=0.90,
    ),
]


# ══════════════════════════════════════════════════════════════════
#  2. THE SHAKESPEARE PROFILE
#     Threshold values a person must meet in each dimension.
#     These are necessary conditions (AND logic), not sufficient ones.
#
#     Why AND?  A figure who has everything except patronage access
#     probably produces work that isn't recognized or preserved.
#     A figure born before commercial theater has nowhere to perform.
#     We do know of talented writers who vanished due to one missing
#     piece — this is the historically defensible assumption.
# ══════════════════════════════════════════════════════════════════

@dataclass
class ShakespeareProfile:
    """
    Minimum requirements across all dimensions.

    Class note: Craftsman / merchant is the "sweet spot."
    Too poor → no grammar-school education.
    Too noble → wouldn't write commercial plays for money.
    Shakespeare's father was a glover and alderman — squarely here.
    """
    # Individual attributes (binary or probability gates)
    must_be_male:             bool  = True
    right_social_class_prob: float  = 0.25    # craftsman + merchant ≈ 25%

    # Structural thresholds (continuous gates)
    min_theater:      float = 0.50
    min_patronage:    float = 0.35
    min_printing:     float = 0.30
    min_stability:    float = 0.40
    min_vernacular:   float = 0.35
    min_literacy_rate: float = 0.18           # literacy floor for audience + ecosystem

    # Intrinsic talent: top 0.1% of the eligible population.
    # This is the hardest parameter to pin down — but as the
    # sensitivity analysis shows, the era-comparison findings are
    # robust to a wide range of assumptions here.
    talent_top_fraction: float = 0.001


PROFILE = ShakespeareProfile()


# ══════════════════════════════════════════════════════════════════
#  3. PROBABILITY ENGINE
#
#  Why analytical rather than individual-level simulation?
#  Shakespeare-level talent is so rare that direct simulation would
#  need to spawn the entire population of England each year to get
#  statistically stable hits.  The analytical approach gives exact
#  rates; Monte Carlo is then used only for era-parameter noise.
#
#  P(Shakespeare) = P(male)
#                 × P(survive childhood)
#                 × P(right class)
#                 × P(literate | class)
#                 × P(urban-adjacent | class)
#                 × P(talent in top X%)
#                 × Π  structural_gate(condition)
#
#  Factors are treated as conditionally independent given era params.
#  (A necessary simplification — in reality correlations exist.)
# ══════════════════════════════════════════════════════════════════

def structural_gate(actual: float, required: float, penalty: float = 0.05) -> float:
    """
    Soft gate: below-threshold conditions don't make genius impossible —
    just far less likely.

    Returns 1.0 if the condition is met, a small penalty value otherwise.
    The penalty scales with how far below threshold the condition falls.

    A hard binary gate would be wrong: a slightly-below-threshold patronage
    environment makes a Shakespeare figure 20× less likely, not impossible.
    """
    if actual >= required:
        return 1.0
    deficit = (required - actual) / required
    return max(penalty * (1.0 - deficit), 0.001)


def compute_p_shakespeare(
    era:       Era,
    year:      int,
    profile:   ShakespeareProfile,
    noise_std: float = 0.05,
    rng:       np.random.Generator = None,
) -> float:
    """
    Probability that any single person born in `year` under `era` conditions
    becomes a Shakespeare-level figure.

    `noise_std` adds Gaussian noise to structural parameters to simulate
    year-to-year variation within an era (this is where Monte Carlo lives).
    """
    if rng is None:
        rng = np.random.default_rng()

    # ── Structural params with within-era noise ─────────────────
    def noisy(val: float) -> float:
        return float(np.clip(val + rng.normal(0, noise_std), 0.0, 1.0))

    theater    = noisy(era.theater_infrastructure)
    patronage  = noisy(era.patronage_robustness)
    printing   = noisy(era.printing_access)
    stability  = noisy(era.political_stability)
    vernacular = noisy(era.vernacular_culture)
    literacy   = noisy(era.male_literacy_rate)

    # ── Survival (with plague penalty) ─────────────────────────
    survival = era.childhood_survival * (0.72 if year in era.plague_years else 1.0)

    # ── Individual-attribute probabilities ──────────────────────
    p_male     = 0.50
    p_survive  = survival
    p_class    = profile.right_social_class_prob

    # Craftsman/merchant families: above-average literacy and urban presence
    p_literate = min(literacy * 1.20, 1.0)
    p_urban    = min(era.urban_fraction * 1.90, 1.0)

    # Talent: independent of social circumstances, drawn from right tail
    p_talent   = profile.talent_top_fraction

    # ── Structural gates ────────────────────────────────────────
    g_theater    = structural_gate(theater,   profile.min_theater)
    g_patronage  = structural_gate(patronage, profile.min_patronage)
    g_printing   = structural_gate(printing,  profile.min_printing)
    g_stability  = structural_gate(stability, profile.min_stability)
    g_vernacular = structural_gate(vernacular, profile.min_vernacular)
    g_literacy   = structural_gate(literacy,  profile.min_literacy_rate)

    # ── Joint probability ────────────────────────────────────────
    return (
        p_male * p_survive * p_class * p_literate * p_urban * p_talent
        * g_theater * g_patronage * g_printing
        * g_stability * g_vernacular * g_literacy
    )


# ══════════════════════════════════════════════════════════════════
#  4. SIMULATION ENGINE
#
#  For each year in the range:
#    - Get era parameters
#    - Compute P(Shakespeare) per person (with noise)
#    - Expected count = P × population
#
#  Repeat N_MC times → confidence bands on the time series.
#  Aggregate per era → expected total figures per historical period.
# ══════════════════════════════════════════════════════════════════

def get_era(year: int) -> Era:
    for era in ERAS:
        if era.year_start <= year < era.year_end:
            return era
    return ERAS[-1]


def run_simulation(
    year_start: int              = YEAR_START,
    year_end:   int              = YEAR_END,
    n_mc:       int              = 200,
    noise_std:  float            = 0.05,
    profile:    ShakespeareProfile = None,
    base_seed:  int              = 42,
) -> Dict:
    """
    Full Monte Carlo simulation.

    Returns a results dict containing:
      - Per-year time series (mean, std, 10th/90th percentile)
      - Per-era summary statistics
      - Bottleneck decomposition
    """
    if profile is None:
        profile = ShakespeareProfile()

    years   = list(range(year_start, year_end))
    n_years = len(years)

    # mc_counts[trial, year_index] = expected Shakespeare count in that year
    mc_counts = np.zeros((n_mc, n_years))

    for trial in range(n_mc):
        rng = np.random.default_rng(base_seed + trial)
        for yi, year in enumerate(years):
            era = get_era(year)
            p   = compute_p_shakespeare(era, year, profile, noise_std, rng)
            mc_counts[trial, yi] = p * era.population   # expected count = P × population

    mean_count = mc_counts.mean(axis=0)
    p10_count  = np.percentile(mc_counts, 10, axis=0)
    p90_count  = np.percentile(mc_counts, 90, axis=0)

    # ── Per-era summaries ────────────────────────────────────────
    era_summaries = {}
    for era in ERAS:
        idx = [i for i, y in enumerate(years) if era.year_start <= y < era.year_end]
        if not idx:
            continue
        era_summaries[era.name] = {
            'expected_per_year': mean_count[idx].mean(),
            'expected_total':    mean_count[idx].sum(),
            'era_length':        era.year_end - era.year_start,
            'population':        era.population,
        }

    return {
        'years':         years,
        'mean_count':    mean_count,
        'p10_count':     p10_count,
        'p90_count':     p90_count,
        'mc_counts':     mc_counts,
        'era_summaries': era_summaries,
        'bottleneck':    compute_bottlenecks(profile),
    }


def compute_bottlenecks(profile: ShakespeareProfile) -> Dict:
    """
    For each era, decompose how much each structural gate reduces
    P(Shakespeare).  Identifies *which* missing conditions explain
    why certain eras produce fewer Shakespeare-level figures.
    """
    results = {}
    for era in ERAS:
        baseline = (
            0.50
            * era.childhood_survival
            * profile.right_social_class_prob
            * min(era.male_literacy_rate * 1.2, 1.0)
            * min(era.urban_fraction * 1.9, 1.0)
            * profile.talent_top_fraction
        )
        gates = {
            'Theater':    structural_gate(era.theater_infrastructure, profile.min_theater),
            'Patronage':  structural_gate(era.patronage_robustness,   profile.min_patronage),
            'Printing':   structural_gate(era.printing_access,        profile.min_printing),
            'Stability':  structural_gate(era.political_stability,    profile.min_stability),
            'Vernacular': structural_gate(era.vernacular_culture,     profile.min_vernacular),
            'Literacy':   structural_gate(era.male_literacy_rate,     profile.min_literacy_rate),
        }
        multiplier = 1.0
        for g in gates.values():
            multiplier *= g
        results[era.name] = {
            'gates':            gates,
            'total_multiplier': multiplier,
            'baseline_p':       baseline,
            'effective_p':      baseline * multiplier,
        }
    return results


def sensitivity_analysis(
    talent_fractions: List[float],
    profile:          ShakespeareProfile,
    n_mc:             int = 80,
) -> Dict[float, Dict]:
    """
    Run the simulation for multiple talent-threshold assumptions.
    Tests whether the era-comparison findings are robust to
    our most contested parameter.
    """
    results = {}
    for tf in talent_fractions:
        import dataclasses
        modified = dataclasses.replace(profile, talent_top_fraction=tf)
        results[tf] = run_simulation(profile=modified, n_mc=n_mc)
    return results


# ══════════════════════════════════════════════════════════════════
#  5. VISUALIZATION  (5-panel figure)
#
#  Panel 1: Expected count over time with confidence bands
#  Panel 2: Total expected per era (horizontal bar)
#  Panel 3: Structural conditions — Late Medieval vs. Elizabethan
#  Panel 4: Combined gate multiplier per era (the bottleneck index)
#  Panel 5: Sensitivity to talent-threshold assumption
# ══════════════════════════════════════════════════════════════════

ERA_PALETTE = [
    '#e8d5b7','#d5e8b7','#b7d5e8','#e8b7d5',
    '#d5b7e8','#b7e8d5','#e8e8b7','#b7b7e8',
]

def plot_results(results: Dict, profile: ShakespeareProfile):
    years     = results['years']
    mean      = results['mean_count']
    p10       = results['p10_count']
    p90       = results['p90_count']
    summaries = results['era_summaries']
    btk       = results['bottleneck']

    era_names   = list(summaries.keys())
    era_totals  = [summaries[n]['expected_total'] for n in era_names]

    fig = plt.figure(figsize=(17, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.38)

    # ── Panel 1: Time series ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(years, p10, p90, alpha=0.22, color='steelblue',
                     label='10–90th pct. range')
    ax1.plot(years, mean, color='#1a3a6b', linewidth=2.2,
             label='Mean expected count')

    for i, era in enumerate(ERAS):
        ax1.axvspan(era.year_start, era.year_end, alpha=0.12,
                    color=ERA_PALETTE[i % len(ERA_PALETTE)], zorder=0)
        mid = (era.year_start + era.year_end) / 2
        # Label at top of axis
        ymax = ax1.get_ylim()[1] if ax1.get_ylim()[1] != 1.0 else max(p90) * 1.05
        ax1.text(mid, max(p90) * 0.95, era.name,
                 fontsize=7.5, ha='center', va='top', rotation=38, color='#333')

    ax1.axvline(1564, color='crimson', lw=2.0, ls='--',
                label='Shakespeare born (1564)')
    ax1.set_xlim(YEAR_START, YEAR_END)
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Expected Shakespeare-level\nfigures per year (England)', fontsize=10)
    ax1.set_title(
        'Expected Annual Output of Shakespeare-Level Figures, England 1400–1800',
        fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')

    # ── Panel 2: Total per era ───────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    colors_bar = ['#c0392b' if 'Elizabethan' in n else '#3498db' for n in era_names]
    y_pos = range(len(era_names))
    ax2.barh(y_pos, era_totals, color=colors_bar, edgecolor='white', height=0.65)
    ax2.axvline(1.0, color='crimson', lw=1.3, ls='--', alpha=0.7,
                label='= 1 figure')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(era_names, fontsize=8.5)
    ax2.set_xlabel('Total expected Shakespeare-level figures\nacross the entire era', fontsize=10)
    ax2.set_title('Expected Total per Era\n(red = Elizabethan peak)', fontsize=11,
                  fontweight='bold')
    ax2.legend(fontsize=9)

    # ── Panel 3: Structural conditions comparison ────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    gate_labels = ['Theater', 'Patronage', 'Printing', 'Stability', 'Vernacular', 'Literacy']
    med_era   = next(e for e in ERAS if 'Late Medieval' in e.name)
    eliz_era  = next(e for e in ERAS if 'Elizabethan'  in e.name)

    def era_gate_vals(era: Era) -> List[float]:
        return [
            era.theater_infrastructure,
            era.patronage_robustness,
            era.printing_access,
            era.political_stability,
            era.vernacular_culture,
            era.male_literacy_rate,
        ]

    thresholds_list = [
        profile.min_theater, profile.min_patronage, profile.min_printing,
        profile.min_stability, profile.min_vernacular, profile.min_literacy_rate,
    ]

    x, w = np.arange(len(gate_labels)), 0.30
    ax3.bar(x - w, era_gate_vals(med_era),  w, label='Late Medieval',
            color='#7f8c8d', edgecolor='white')
    ax3.bar(x,     era_gate_vals(eliz_era), w, label='Elizabethan',
            color='#c0392b', edgecolor='white')
    for xi, tv in zip(x - w/2, thresholds_list):
        ax3.plot([xi - 0.38, xi + 0.38], [tv, tv], 'k:', lw=1.8)

    ax3.set_xticks(x - w/2)
    ax3.set_xticklabels(gate_labels, fontsize=9.5)
    ax3.set_ylabel('Score (0–1)', fontsize=10)
    ax3.set_ylim(0, 1.08)
    ax3.set_title('Structural Conditions: Medieval vs. Elizabethan\n'
                  '(dotted = required threshold)', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)

    # ── Panel 4: Gate multiplier (bottleneck index) ──────────────
    ax4 = fig.add_subplot(gs[2, 0])
    b_names      = list(btk.keys())
    multipliers  = [btk[n]['total_multiplier'] for n in b_names]
    colors_btk   = ['#c0392b' if 'Elizabethan' in n else '#3498db' for n in b_names]
    ax4.bar(range(len(b_names)), multipliers, color=colors_btk, edgecolor='white')
    ax4.set_xticks(range(len(b_names)))
    ax4.set_xticklabels(b_names, rotation=36, ha='right', fontsize=8.5)
    ax4.set_ylabel('Combined gate multiplier (0–1)', fontsize=10)
    ax4.set_ylim(0, 1.08)
    ax4.set_title('Structural Bottleneck Index\n'
                  'How much do era conditions reduce P(Shakespeare)?',
                  fontsize=11, fontweight='bold')
    ax4.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)

    # ── Panel 5: Sensitivity to talent threshold ─────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    talent_fracs = [0.005, 0.002, 0.001, 0.0005, 0.0002]
    print('  Running sensitivity analysis across talent thresholds...')
    sens_results = sensitivity_analysis(talent_fracs, profile, n_mc=60)

    for tf, sr in sens_results.items():
        sr_era_names   = list(sr['era_summaries'].keys())
        sr_era_totals  = [sr['era_summaries'][n]['expected_total'] for n in sr_era_names]
        ax5.plot(range(len(sr_era_names)), sr_era_totals,
                 marker='o', markersize=5,
                 label=f"top {tf * 100:.2f}%", alpha=0.75)

    ax5.set_xticks(range(len(era_names)))
    ax5.set_xticklabels(era_names, rotation=36, ha='right', fontsize=8.5)
    ax5.set_ylabel('Expected total figures per era', fontsize=10)
    ax5.set_title('Sensitivity to Talent-Threshold Assumption\n'
                  '(era rankings are robust regardless)', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=8.5, title='Talent threshold', ncol=2)

    fig.suptitle(
        'Monte Carlo Simulation: The Conditions for a Shakespeare  (England, 1400–1800)',
        fontsize=13, fontweight='bold', y=1.005)

    plt.savefig('shakespeare_simulation.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.show()
    print('\n  Figure saved → shakespeare_simulation.png')


# ══════════════════════════════════════════════════════════════════
#  6. TEXT SUMMARY
# ══════════════════════════════════════════════════════════════════

def print_summary(results: Dict):
    summaries = results['era_summaries']
    btk       = results['bottleneck']
    eliz_tot  = summaries.get('Elizabethan', {}).get('expected_total', 1.0)

    print('\n' + '═' * 65)
    print('   SHAKESPEARE EMERGENCE SIMULATION — RESULTS')
    print('═' * 65)

    for name, vals in summaries.items():
        ratio  = vals['expected_total'] / eliz_tot if eliz_tot > 0 else 0
        mult   = btk.get(name, {}).get('total_multiplier', 0)
        eff_p  = btk.get(name, {}).get('effective_p', 0)
        print(f"\n  {name}  ({vals['era_length']} yrs, pop. {vals['population']:,})")
        print(f"    P(Shakespeare) per person:   {eff_p:.3e}")
        print(f"    Expected figures per year:   {vals['expected_per_year']:.4f}")
        print(f"    Expected total for era:      {vals['expected_total']:.3f}")
        print(f"    vs. Elizabethan:             {ratio:.3f}×")
        print(f"    Gate multiplier:             {mult:.5f}")

    med  = summaries.get('Late Medieval', {})
    eliz = summaries.get('Elizabethan', {})
    if med and eliz and med['expected_total'] > 0:
        ratio = eliz['expected_total'] / med['expected_total']
        print('\n' + '═' * 65)
        print('   KEY FINDING')
        print('═' * 65)
        print(f"\n   The Elizabethan era was {ratio:.0f}× more likely than the Late")
        print("   Medieval period to produce a Shakespeare-level figure —")
        print("   not because individuals were more talented, but because the")
        print("   structural gates (theater, patronage, printing, literacy)")
        print("   were all simultaneously cleared for the first time.")

    print('\n   Note: "expected total < 1.0" does not mean a Shakespeare')
    print('   could not emerge — it means the expected emergence interval')
    print('   exceeds the era length.  The Elizabethan window is uniquely')
    print('   compressed: many gates open at once, for the first time.')
    print('═' * 65 + '\n')


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print()
    print('  Shakespeare Emergence Simulation')
    print('  ─────────────────────────────────')
    print(f'  Years:           {YEAR_START}–{YEAR_END}')
    print(f'  Monte Carlo runs: 200')
    print(f'  Talent threshold: top {PROFILE.talent_top_fraction * 100:.1f}%')
    print(f'  Within-era noise: σ = 0.05')
    print()

    results = run_simulation(
        year_start=YEAR_START,
        year_end=YEAR_END,
        n_mc=200,
        noise_std=0.05,
        profile=PROFILE,
        base_seed=42,
    )

    print_summary(results)
    plot_results(results, PROFILE)
