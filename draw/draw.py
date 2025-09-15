import matplotlib.pyplot as plt
import numpy as np

# Set style for better aesthetics
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['text.color'] = '#333333'

# Sample data based on your description
steps = np.arange(1, 16)

# Success Rate (SR) data - StepNav converges quickly, FlowNav slower
stepnav_sr = np.array([82, 85, 88, 92, 95, 96, 96.5, 97, 97, 97, 97, 97, 97, 97, 97])
flownav_sr = np.array([80, 82, 84, 86, 88, 89, 90, 91, 92, 93, 94, 95, 96, 96.5, 97])
navibridger_sr = np.array([81, 83, 86, 89, 91, 92, 93, 94, 95, 95.5, 96, 96, 96.5, 96.5, 97])

# Collision rate (Coll.) data - lower is better, should decrease with steps
stepnav_coll = np.array([0.95, 0.85, 0.75, 0.65, 0.58, 0.55, 0.54, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53])
flownav_coll = np.array([0.98, 0.92, 0.87, 0.82, 0.78, 0.75, 0.72, 0.69, 0.66, 0.63, 0.60, 0.58, 0.56, 0.55, 0.54])
navibridger_coll = np.array([0.96, 0.90, 0.84, 0.78, 0.72, 0.68, 0.65, 0.62, 0.60, 0.58, 0.57, 0.56, 0.55, 0.55, 0.54])

# Modern color palette
colors = {
    'stepnav': '#2E86AB',      # Ocean blue
    'flownav': '#A23B72',      # Burgundy
    'navibridger': '#F18F01'   # Orange
}

# Create figure and axis with better styling
fig, ax1 = plt.subplots(figsize=(12, 7), facecolor='#F8F9FA')
ax1.set_facecolor('#F8F9FA')

# Plot SR on left y-axis with improved styling
ax1.set_xlabel('Refinement Steps', fontsize=16, fontweight='bold', color='#333333')
ax1.set_ylabel('Success Rate (%)', fontsize=16, fontweight='bold', color=colors['stepnav'])
ax1.plot(steps, stepnav_sr, 'o-', color=colors['stepnav'], label='StepNav', 
         linewidth=3, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax1.plot(steps, flownav_sr, 's-', color=colors['flownav'], label='FlowNav', 
         linewidth=3, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax1.plot(steps, navibridger_sr, '^-', color=colors['navibridger'], label='NaviBridger', 
         linewidth=3, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax1.tick_params(axis='y', labelcolor=colors['stepnav'], labelsize=12)
ax1.set_ylim(79, 98)
ax1.set_xlim(0.5, 15.5)

# Enhanced grid
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, color='#666666')
ax1.set_axisbelow(True)

# Create second y-axis for Collision rate with improved styling
ax2 = ax1.twinx()
ax2.set_ylabel('Collision Rate', fontsize=16, fontweight='bold', color=colors['flownav'])
ax2.plot(steps, stepnav_coll, 'o--', color=colors['stepnav'], alpha=0.8, 
         linewidth=2.5, markersize=6, markerfacecolor='white', markeredgewidth=1.5)
ax2.plot(steps, flownav_coll, 's--', color=colors['flownav'], alpha=0.8, 
         linewidth=2.5, markersize=6, markerfacecolor='white', markeredgewidth=1.5)
ax2.plot(steps, navibridger_coll, '^--', color=colors['navibridger'], alpha=0.8, 
         linewidth=2.5, markersize=6, markerfacecolor='white', markeredgewidth=1.5)
ax2.tick_params(axis='y', labelcolor=colors['flownav'], labelsize=14)
ax2.set_ylim(0.52, 1.02)

# Enhanced legend
lines1, labels1 = ax1.get_legend_handles_labels()
# legend = ax1.legend(lines1, labels1, loc='upper left', bbox_to_anchor=(0.02, 0.98), 
#                     fontsize=12, frameon=True, fancybox=True, shadow=True, 
#                     facecolor='white', edgecolor='#CCCCCC')
legend = ax1.legend(lines1, labels1, loc='center right', 
                    fontsize=16, frameon=True, fancybox=True, shadow=True, 
                    facecolor='white', edgecolor='#CCCCCC')
legend.get_frame().set_linewidth(1.5)

# Add informative annotations
# ax1.text(0.02, 0.98, 'Solid lines: Success Rate', transform=ax1.transAxes, fontsize=14, 
#          verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
#          alpha=0.9, edgecolor='#CCCCCC'), fontweight='bold')
# ax1.text(0.22, 0.98, 'Dashed lines: Collision Rate', transform=ax1.transAxes, fontsize=14,
#          verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
#          alpha=0.9, edgecolor='#CCCCCC'), fontweight='bold')

# Enhanced title
# plt.title('Navigation Performance Scaling with Refinement Steps', fontsize=16, 
        #   fontweight='bold', pad=30, color='#333333')

# Add subtle background pattern
ax1.add_patch(plt.Rectangle((0.5, 79), 15, 19, fill=True, color='#F0F8FF', alpha=0.1, zorder=-1))

plt.tight_layout()


# Optional: Save the figure with high quality
# plt.savefig('fig_steps_scaling_beautiful.pdf', dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.savefig('fig_steps_scaling_beautiful.png', dpi=300, bbox_inches='tight', facecolor='#F8F9FA')

plt.show()