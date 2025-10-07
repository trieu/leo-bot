import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Define phases and tasks
phases = {
    "Phase 1 – Foundation": [
        "leo-cdp-framework (core infra)",
        "C720 Product Documentation (baseline)",
        "demo-app-leocdp (skeleton)"
    ],
    "Phase 2 – Data Core": [
        "C720 Flutter SDK (mobile data collection)",
        "Profile ingestion in leo-cdp-framework",
        "C720 Data Enrichment Agents (MVP)"
    ],
    "Phase 3 – AI Engines": [
        "C720 Data Segmentation Agents",
        "C720 Recommendation Engine",
        "Feedback loop into master profiles"
    ],
    "Phase 4 – Engagement Layer": [
        "C720 Marketing Automation",
        "leo-chatbot",
        "Cross-channel personalization workflows"
    ],
    "Phase 5 – Demo & Scaling": [
        "demo-app-leocdp (full showcase)",
        "Expanded Product Documentation",
        "leo-cdp-framework (self-hosted + cloud)"
    ]
}

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")

# Vertical spacing
y = 1.0
dy = -0.15

# Colors for phases
colors = ["#cce5ff", "#d4edda", "#fff3cd", "#f8d7da", "#e2e3e5"]

# Draw phases with tasks
for i, (phase, tasks) in enumerate(phases.items()):
    # Draw phase box
    ax.text(0.0, y, phase, fontsize=8, weight="bold", ha="left",
            bbox=dict(facecolor=colors[i], edgecolor="black", boxstyle="round,pad=0.3"))
    y += dy
    # Draw tasks under phase
    for task in tasks:
        ax.text(0.05, y, f"- {task}", fontsize=10, ha="left")
        y += dy
    y += dy  # extra space between phases

plt.tight_layout()
output_path = "c720_ai_cdp_roadmap_matplotlib.png"
plt.savefig(output_path, dpi=150)
plt.close()

output_path
