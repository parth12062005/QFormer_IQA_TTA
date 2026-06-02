import matplotlib.pyplot.subplots
import matplotlib.pyplot as plt

# Data extracted from log
batches = list(range(1, 31))
diffs = [0.031319, 0.027364, 0.041844, 0.060668, 0.079838, 0.099614, 0.113580, 0.149722, 0.173486, 0.182108, 0.212186, 0.222392, 0.248762, 0.281619, 0.304995, 0.330192, 0.329469, 0.351204, 0.391807, 0.380653, 0.422257, 0.423922, 0.463538, 0.469618, 0.478774, 0.481577, 0.517577, 0.510931, 0.544240, 0.553505]
cos_sims = [0.1341, -0.0136, -0.0712, -0.0669, 0.1080, 0.2050, -0.0756, 0.1001, 0.0467, 0.2068, 0.2034, 0.1591, -0.0163, -0.0840, 0.1512, 0.0182, 0.0781, 0.1617, 0.2032, 0.0401, 0.1586, 0.0905, 0.0899, 0.0122, 0.1671, 0.1317, -0.0325, 0.1733, 0.0590, -0.0511]

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Batch (1-30)', fontweight='bold')
ax1.set_ylabel('Cosine Similarity (TTA vs Oracle)', color=color, fontweight='bold')
ax1.plot(batches, cos_sims, marker='o', color=color, label='Cosine Similarity')
ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero Similarity (Orthogonal)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(-0.2, 0.3)

ax2 = ax1.twinx()
color = 'tab:gray'
ax2.set_ylabel('Oracle Gradient Drift (L2 Diff)', color=color)
ax2.plot(batches, diffs, marker='x', linestyle=':', color=color, label='Weight Drift Factor')
ax2.tick_params(axis='y', labelcolor=color)

fig.suptitle('Gradient Alignment: Unsupervised TTA vs Supervised Oracle', fontsize=14, fontweight='bold')
fig.tight_layout()

# Add a text box with the average
avg_sim = sum(cos_sims)/len(cos_sims)
plt.text(0.02, 0.95, f'Average Cosine Sim: {avg_sim:.4f}\n(Near 0 = Orthogonal/No Alignment)', 
         transform=ax1.transAxes, fontsize=12, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig('/home/parth/.gemini/antigravity/brain/3db10c70-c690-49e2-a7fb-ec92cdc446a0/gradient_alignment.png')
print("Plot saved.")
