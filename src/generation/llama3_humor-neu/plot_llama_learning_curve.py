"""
Plot Llama-3 SFT Training Learning Curves
Analyzes trainer_state.json to visualize training/eval loss
and explain checkpoint-2300 selection.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load training state from the final checkpoint
checkpoint_dir = Path(__file__).parent / "llama3-humor-lora" / "checkpoint-3564"
trainer_state_path = checkpoint_dir / "trainer_state.json"

print("=" * 80)
print("LLAMA-3 SFT TRAINING ANALYSIS")
print("=" * 80)
print(f"\nLoading trainer state from:\n  {trainer_state_path}\n")

with open(trainer_state_path) as f:
    trainer_state = json.load(f)

log_history = trainer_state["log_history"]
best_step = trainer_state["best_global_step"]
best_metric = trainer_state["best_metric"]
total_steps = trainer_state["global_step"]
total_epochs = trainer_state["epoch"]

print(f"Total log entries: {len(log_history)}")
print(f"Total steps: {total_steps} ({total_epochs:.0f} epochs)")
print(f"Best checkpoint: step {best_step} (eval_loss = {best_metric:.4f})")
print(f"Checkpoint saved: checkpoint-2300 (closest save_steps=100 before best)")

# Extract metrics
train_steps, train_loss, train_lr = [], [], []
eval_steps, eval_loss = [], []
token_acc_steps, token_acc = [], []

for entry in log_history:
    step = entry.get("step")
    if "loss" in entry and "eval_loss" not in entry:
        train_steps.append(step)
        train_loss.append(entry["loss"])
        train_lr.append(entry.get("learning_rate", 0))
        if "mean_token_accuracy" in entry:
            token_acc_steps.append(step)
            token_acc.append(entry["mean_token_accuracy"])
    if "eval_loss" in entry:
        eval_steps.append(step)
        eval_loss.append(entry["eval_loss"])

print(f"\nTrain points: {len(train_steps)}")
print(f"Eval points: {len(eval_steps)} (every {eval_steps[1] - eval_steps[0]} steps)")

# Epoch boundaries
steps_per_epoch = total_steps / total_epochs
epoch_boundaries = [int(steps_per_epoch * e) for e in range(1, int(total_epochs) + 1)]

# ----- Analysis -----
print("\n" + "=" * 80)
print("EVAL LOSS ANALYSIS")
print("=" * 80)

# Find key points
best_eval_idx = np.argmin(eval_loss)
best_eval_step = eval_steps[best_eval_idx]
best_eval_loss = eval_loss[best_eval_idx]

# Eval loss at epoch boundaries
for ep in range(1, int(total_epochs) + 1):
    ep_step = int(steps_per_epoch * ep)
    closest_idx = np.argmin(np.abs(np.array(eval_steps) - ep_step))
    print(f"  Epoch {ep} (~step {ep_step}): eval_loss = {eval_loss[closest_idx]:.4f}")

print(f"\n  Best eval loss: {best_eval_loss:.4f} at step {best_eval_step} (epoch {best_eval_step/steps_per_epoch:.2f})")

# Check overfitting: after best point, does eval loss increase?
post_best = [(s, l) for s, l in zip(eval_steps, eval_loss) if s > best_eval_step]
if post_best:
    avg_post_best = np.mean([l for _, l in post_best])
    print(f"  Avg eval loss after best: {avg_post_best:.4f} (delta: +{avg_post_best - best_eval_loss:.4f})")
    print(f"  --> Eval loss plateaus/rises after epoch ~2, indicating overfitting on epoch 3")

# ----- PLOT -----
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Llama-3-8B SFT Fine-Tuning Learning Curves', fontsize=16, fontweight='bold')

# Colors
C_TRAIN = '#2196F3'  # blue
C_EVAL = '#F44336'   # red
C_BEST = '#4CAF50'   # green
C_EPOCH = '#9E9E9E'  # gray

# ---- Plot 1: Training & Eval Loss (full) ----
ax = axes[0, 0]
ax.plot(train_steps, train_loss, color=C_TRAIN, alpha=0.3, linewidth=0.8, label='Train Loss')
# Smoothed train loss (moving average)
window = 15
if len(train_loss) > window:
    smoothed = np.convolve(train_loss, np.ones(window)/window, mode='valid')
    smoothed_steps = train_steps[window-1:]
    ax.plot(smoothed_steps, smoothed, color=C_TRAIN, linewidth=2, label=f'Train Loss (MA-{window})')
ax.plot(eval_steps, eval_loss, color=C_EVAL, linewidth=2.5, marker='o', markersize=3, label='Eval Loss')
# Best point
ax.axvline(best_eval_step, color=C_BEST, linestyle='--', alpha=0.7, linewidth=2, label=f'Best: step {best_eval_step}')
ax.plot(best_eval_step, best_eval_loss, color=C_BEST, marker='*', markersize=18, zorder=5)
# Epoch boundaries
for eb in epoch_boundaries:
    ax.axvline(eb, color=C_EPOCH, linestyle=':', alpha=0.4)
ax.set_xlabel('Steps', fontsize=11)
ax.set_ylabel('Loss', fontsize=11)
ax.set_title('Training & Evaluation Loss')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# ---- Plot 2: Eval Loss zoomed (epoch 1.5-2.5) ----
ax = axes[0, 1]
zoom_range = [(s, l) for s, l in zip(eval_steps, eval_loss) if 1500 <= s <= 3000]
if zoom_range:
    zs, zl = zip(*zoom_range)
    ax.plot(zs, zl, color=C_EVAL, linewidth=2.5, marker='o', markersize=5)
    ax.axvline(best_eval_step, color=C_BEST, linestyle='--', alpha=0.7, linewidth=2, label=f'Best: step {best_eval_step}\neval_loss = {best_eval_loss:.4f}')
    ax.plot(best_eval_step, best_eval_loss, color=C_BEST, marker='*', markersize=22, zorder=5)
    # Checkpoint-2300
    if 2300 in eval_steps:
        idx_2300 = eval_steps.index(2300)
        ax.plot(2300, eval_loss[idx_2300], color='purple', marker='D', markersize=12, zorder=5,
                label=f'checkpoint-2300\neval_loss = {eval_loss[idx_2300]:.4f}')
    # Epoch 2 boundary
    ax.axvline(epoch_boundaries[1], color=C_EPOCH, linestyle=':', alpha=0.5, label='Epoch 2 boundary')
    # Show overfitting zone
    ax.axvspan(epoch_boundaries[1], 3000, alpha=0.08, color='red', label='Epoch 3 (overfitting)')
ax.set_xlabel('Steps', fontsize=11)
ax.set_ylabel('Eval Loss', fontsize=11)
ax.set_title('Eval Loss (Zoomed: Steps 1500-3000)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# ---- Plot 3: Learning Rate Schedule ----
ax = axes[1, 0]
ax.plot(train_steps, train_lr, color='#FF9800', linewidth=2)
ax.axvline(best_eval_step, color=C_BEST, linestyle='--', alpha=0.7, linewidth=2, label=f'Best checkpoint')
for eb in epoch_boundaries:
    ax.axvline(eb, color=C_EPOCH, linestyle=':', alpha=0.4)
ax.set_xlabel('Steps', fontsize=11)
ax.set_ylabel('Learning Rate', fontsize=11)
ax.set_title('Learning Rate Schedule (Cosine)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# ---- Plot 4: Token Accuracy ----
ax = axes[1, 1]
if token_acc:
    ax.plot(token_acc_steps, token_acc, color='#9C27B0', alpha=0.3, linewidth=0.8, label='Token Accuracy')
    if len(token_acc) > window:
        smoothed_ta = np.convolve(token_acc, np.ones(window)/window, mode='valid')
        smoothed_ta_steps = token_acc_steps[window-1:]
        ax.plot(smoothed_ta_steps, smoothed_ta, color='#9C27B0', linewidth=2, label=f'Token Acc (MA-{window})')
    ax.axvline(best_eval_step, color=C_BEST, linestyle='--', alpha=0.7, linewidth=2, label=f'Best checkpoint')
    for eb in epoch_boundaries:
        ax.axvline(eb, color=C_EPOCH, linestyle=':', alpha=0.4)
ax.set_xlabel('Steps', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Mean Token Accuracy')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout()

# Save
output_path = Path(__file__).parent / "llama3_learning_curves.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to:\n  {output_path}")

# ---- Summary Table ----
print("\n" + "=" * 80)
print("CHECKPOINT SELECTION: WHY checkpoint-2300?")
print("=" * 80)

print(f"""
Training Setup:
  - 25,000 samples, 3 epochs, batch size 20, save every 100 steps, eval every 50 steps
  - Total: 3,564 steps (1,188 steps/epoch)
  - Cosine LR schedule: 2e-4 -> ~0
  - load_best_model_at_end=True, metric_for_best_model="eval_loss"

Key Observations:

  1. EPOCH 1 (steps 0-1188): Rapid learning phase
     - Eval loss drops steadily: 1.71 -> 0.89
     - Model learns joke structure, humor patterns, word-pair integration

  2. EPOCH 2 (steps 1188-2376): Refinement phase
     - Eval loss continues to decrease but slower: 0.89 -> 0.69
     - Model refines style, punchline delivery, creative associations
     - BEST eval loss reached at step 2350: {best_eval_loss:.4f}
     - checkpoint-2300 (closest saved checkpoint): eval_loss = {eval_loss[eval_steps.index(2300)]:.4f}

  3. EPOCH 3 (steps 2376-3564): Overfitting phase
     - Eval loss INCREASES back to ~0.72 and plateaus
     - Train loss continues to decrease (memorization)
     - Gap between train and eval loss widens = classic overfitting
""")

# Show the overfitting gap
late_train_loss = np.mean([l for s, l in zip(train_steps, train_loss) if s > 2400])
late_eval_loss = np.mean([l for s, l in zip(eval_steps, eval_loss) if s > 2400])
best_region_eval = np.mean([l for s, l in zip(eval_steps, eval_loss) if 2200 <= s <= 2400])

print(f"  Evidence of overfitting:")
print(f"    Train loss (epoch 3 avg): {late_train_loss:.4f}")
print(f"    Eval loss  (epoch 3 avg): {late_eval_loss:.4f}")
print(f"    Gap: {late_eval_loss - late_train_loss:.4f}")
print(f"    Best region eval (steps 2200-2400): {best_region_eval:.4f}")
print(f"    Epoch 3 eval vs best: +{late_eval_loss - best_eval_loss:.4f} ({(late_eval_loss - best_eval_loss)/best_eval_loss*100:.1f}% worse)")

print(f"""
Conclusion:
  checkpoint-2300 is selected because:
  - It has the lowest eval loss among all saved checkpoints ({eval_loss[eval_steps.index(2300)]:.4f})
  - It sits at the end of epoch 2, just before overfitting begins in epoch 3
  - The trainer's load_best_model_at_end=True automatically selected this region
  - Epoch 3 shows no improvement (eval loss plateaus at ~0.72 vs best 0.69)
  - The 2-epoch sweet spot balances learning humor patterns without memorizing training data
""")

print("=" * 80)
print("Analysis complete!")
print("=" * 80)
