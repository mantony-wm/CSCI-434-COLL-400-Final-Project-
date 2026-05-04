"""
Statistical Analysis — Feature Comparison Across Labels
========================================================
Performs comprehensive statistical tests to compare features
between Amazon, Hulu, and YouTube traffic.

Tests included:
- T-test (parametric)
- Mann-Whitney U (non-parametric)
- Cohen's D (effect size)
- ANOVA (all 3 groups)
- Kruskal-Wallis (non-parametric ANOVA)
- Overlap percentage (mean ± 1σ)

Run:
    python statistical_tests.py
    
Outputs: Prints all test results to console
"""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('combined_second_flows.csv')

FEATURES = ['Packet_Count','Total_Length','Average_Packet_Interval','Maximum_Packet_Interval',
            'Minimum_Packet_Interval','Average_Packet_Length','Maximum_Packet_Length',
            'Minimum_Packet_Length','Most_Common_Packet_Length']

amazon  = df[df['Label']=='amazon']
hulu    = df[df['Label']=='hulu']
youtube = df[df['Label']=='youtube']

print("="*80)
print("DATASET SUMMARY")
print("="*80)
print(f"Total windows: {len(df)}")
print(f"Amazon:  {len(amazon)} windows")
print(f"Hulu:    {len(hulu)} windows")
print(f"YouTube: {len(youtube)} windows\n")

# ========== Cohen's D Effect Size ==========
def cohens_d(x1, x2):
    """Calculate Cohen's D effect size between two groups"""
    n1, n2 = len(x1), len(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
    return (np.mean(x1) - np.mean(x2)) / pooled_std if pooled_std > 0 else 0

# ========== T-TEST (Independent Samples) ==========
print("="*80)
print("T-TEST (parametric) — p-values")
print("p < 0.001 = highly significant | p < 0.05 = significant | p > 0.05 = NOT significant")
print("="*80)
print(f"{'Feature':<35} {'Am-Hu':<12} {'Am-Yt':<12} {'Hu-Yt':<12}")
print("-"*80)

for feat in FEATURES:
    t_ah, p_ah = stats.ttest_ind(amazon[feat], hulu[feat])
    t_ay, p_ay = stats.ttest_ind(amazon[feat], youtube[feat])
    t_hy, p_hy = stats.ttest_ind(hulu[feat], youtube[feat])
    
    # Flag non-significant results
    sig_ah = "***" if p_ah < 0.001 else ("*" if p_ah < 0.05 else "NS")
    sig_ay = "***" if p_ay < 0.001 else ("*" if p_ay < 0.05 else "NS")
    sig_hy = "***" if p_hy < 0.001 else ("*" if p_hy < 0.05 else "NS")
    
    print(f"{feat:<35} {p_ah:>8.6f} {sig_ah:<3} {p_ay:>8.6f} {sig_ay:<3} {p_hy:>8.6f} {sig_hy:<3}")

# ========== COHEN'S D (Effect Size) ==========
print("\n" + "="*80)
print("COHEN'S D — Effect Size")
print("Small: 0.2 | Medium: 0.5 | Large: 0.8")
print("="*80)
print(f"{'Feature':<35} {'Am-Hu':<12} {'Am-Yt':<12} {'Hu-Yt':<12}")
print("-"*80)

for feat in FEATURES:
    d_ah = cohens_d(amazon[feat], hulu[feat])
    d_ay = cohens_d(amazon[feat], youtube[feat])
    d_hy = cohens_d(hulu[feat], youtube[feat])
    
    # Flag effect sizes
    flag_ah = "Large" if abs(d_ah) > 0.8 else ("Med" if abs(d_ah) > 0.5 else ("Small" if abs(d_ah) > 0.2 else "Neg"))
    flag_ay = "Large" if abs(d_ay) > 0.8 else ("Med" if abs(d_ay) > 0.5 else ("Small" if abs(d_ay) > 0.2 else "Neg"))
    flag_hy = "Large" if abs(d_hy) > 0.8 else ("Med" if abs(d_hy) > 0.5 else ("Small" if abs(d_hy) > 0.2 else "Neg"))
    
    print(f"{feat:<35} {d_ah:>7.4f} {flag_ah:<5} {d_ay:>7.4f} {flag_ay:<5} {d_hy:>7.4f} {flag_hy:<5}")

# ========== MANN-WHITNEY U (Non-parametric) ==========
print("\n" + "="*80)
print("MANN-WHITNEY U TEST (non-parametric) — p-values")
print("Use when distributions are non-normal (traffic data often is)")
print("="*80)
print(f"{'Feature':<35} {'Am-Hu':<12} {'Am-Yt':<12} {'Hu-Yt':<12}")
print("-"*80)

for feat in FEATURES:
    u_ah, p_ah = stats.mannwhitneyu(amazon[feat], hulu[feat], alternative='two-sided')
    u_ay, p_ay = stats.mannwhitneyu(amazon[feat], youtube[feat], alternative='two-sided')
    u_hy, p_hy = stats.mannwhitneyu(hulu[feat], youtube[feat], alternative='two-sided')
    
    sig_ah = "***" if p_ah < 0.001 else ("*" if p_ah < 0.05 else "NS")
    sig_ay = "***" if p_ay < 0.001 else ("*" if p_ay < 0.05 else "NS")
    sig_hy = "***" if p_hy < 0.001 else ("*" if p_hy < 0.05 else "NS")
    
    print(f"{feat:<35} {p_ah:>8.6f} {sig_ah:<3} {p_ay:>8.6f} {sig_ay:<3} {p_hy:>8.6f} {sig_hy:<3}")

# ========== ANOVA (All 3 Groups) ==========
print("\n" + "="*80)
print("ONE-WAY ANOVA — Tests if all 3 groups differ")
print("="*80)
print(f"{'Feature':<35} {'F-statistic':<15} {'p-value':<15}")
print("-"*80)

for feat in FEATURES:
    f_stat, p_val = stats.f_oneway(amazon[feat], hulu[feat], youtube[feat])
    sig = "***" if p_val < 0.001 else ("*" if p_val < 0.05 else "NS")
    print(f"{feat:<35} {f_stat:>12.4f}    {p_val:>10.6f} {sig}")

# ========== KRUSKAL-WALLIS (Non-parametric ANOVA) ==========
print("\n" + "="*80)
print("KRUSKAL-WALLIS H-TEST (non-parametric ANOVA)")
print("="*80)
print(f"{'Feature':<35} {'H-statistic':<15} {'p-value':<15}")
print("-"*80)

for feat in FEATURES:
    h_stat, p_val = stats.kruskal(amazon[feat], hulu[feat], youtube[feat])
    sig = "***" if p_val < 0.001 else ("*" if p_val < 0.05 else "NS")
    print(f"{feat:<35} {h_stat:>12.4f}    {p_val:>10.6f} {sig}")

# ========== OVERLAP PERCENTAGE ==========
print("\n" + "="*80)
print("OVERLAP PERCENTAGE — % of Class A windows in Class B mean±1σ range")
print("High overlap (>85%) indicates classes are hard to separate")
print("="*80)

print("\nHulu windows falling in YouTube's range:")
for feat in FEATURES:
    yt_mean, yt_std = youtube[feat].mean(), youtube[feat].std()
    overlap = ((hulu[feat] >= yt_mean - yt_std) & (hulu[feat] <= yt_mean + yt_std)).mean() * 100
    flag = "HIGH" if overlap > 85 else ("MED" if overlap > 70 else "LOW")
    print(f"  {feat:<33} {overlap:>6.1f}%  [{flag}]")

print("\nAmazon windows falling in Hulu's range:")
for feat in FEATURES:
    hu_mean, hu_std = hulu[feat].mean(), hulu[feat].std()
    overlap = ((amazon[feat] >= hu_mean - hu_std) & (amazon[feat] <= hu_mean + hu_std)).mean() * 100
    flag = "HIGH" if overlap > 85 else ("MED" if overlap > 70 else "LOW")
    print(f"  {feat:<33} {overlap:>6.1f}%  [{flag}]")

print("\nAmazon windows falling in YouTube's range:")
for feat in FEATURES:
    yt_mean, yt_std = youtube[feat].mean(), youtube[feat].std()
    overlap = ((amazon[feat] >= yt_mean - yt_std) & (amazon[feat] <= yt_mean + yt_std)).mean() * 100
    flag = "HIGH" if overlap > 85 else ("MED" if overlap > 70 else "LOW")
    print(f"  {feat:<33} {overlap:>6.1f}%  [{flag}]")

# ========== KEY FINDINGS ==========
print("\n" + "="*80)
print("KEY FINDINGS FOR YOUR REPORT")
print("="*80)

# Find non-significant features for Hulu-YouTube
non_sig_hy = []
for feat in FEATURES:
    t_hy, p_hy = stats.ttest_ind(hulu[feat], youtube[feat])
    if p_hy > 0.05:
        non_sig_hy.append((feat, p_hy))

print(f"\n1. Features with NO significant difference between Hulu and YouTube:")
print(f"   (These are why the models struggle with Hulu-YouTube boundary)")
for feat, p in non_sig_hy:
    print(f"   - {feat}: p = {p:.4f}")

# Find features with large effect size
large_effects = []
for feat in FEATURES:
    d_hy = abs(cohens_d(hulu[feat], youtube[feat]))
    if d_hy > 0.8:
        large_effects.append((feat, d_hy))

if large_effects:
    print(f"\n2. Features with LARGE effect size (Cohen's D > 0.8) for Hulu-YouTube:")
    for feat, d in large_effects:
        print(f"   - {feat}: D = {d:.4f}")
else:
    print(f"\n2. NO features show large effect size (D > 0.8) for Hulu-YouTube separation")
    print(f"   This quantitatively explains the classification difficulty")

print("\n3. Recommendation for discussion:")
print("   - Cite the t-test p-values for non-significant features")
print("   - Cite the Cohen's D values showing weak/negligible effect sizes")
print("   - Use the overlap percentages to show compound overlap problem")
print("   - Compare Hulu-YouTube statistics to Amazon-YouTube statistics")
print("     to show why Amazon is easier to classify\n")
