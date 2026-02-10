"""
Plot Mistral Training Learning Curves
Analyzes trainer_state.json to identify the best checkpoint
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load training state
checkpoint_dir = Path(r"C:\Users\Anwender\humor-project\src\evaluation\mistral_model\checkpoints_ministral3_multitask\checkpoint-5400")
trainer_state_path = checkpoint_dir / "trainer_state.json"

print("="*80)
print("MISTRAL TRAINING ANALYSIS")
print("="*80)
print(f"\nLoading trainer state from:\n  {trainer_state_path}\n")

with open(trainer_state_path) as f:
    trainer_state = json.load(f)

log_history = trainer_state["log_history"]
print(f"Total log entries: {len(log_history)}")

# Extract metrics
train_steps = []
train_loss = []
train_lr = []

eval_steps = []
eval_loss = []
eval_cls_acc = []
eval_cls_f1 = []
eval_reg_mae = []
eval_reg_rmse = []
eval_reg_r2 = []

for entry in log_history:
    step = entry.get("step")

    # Training metrics
    if "loss" in entry and "eval_loss" not in entry:
        train_steps.append(step)
        train_loss.append(entry["loss"])
        train_lr.append(entry.get("learning_rate", 0))

    # Evaluation metrics
    if "eval_loss" in entry:
        eval_steps.append(step)
        eval_loss.append(entry["eval_loss"])
        eval_cls_acc.append(entry.get("eval_cls_acc", 0))
        eval_cls_f1.append(entry.get("eval_cls_f1", 0))
        eval_reg_mae.append(entry.get("eval_reg_mae", 0))
        eval_reg_rmse.append(entry.get("eval_reg_rmse", 0))
        eval_reg_r2.append(entry.get("eval_reg_r2", 0))

print(f"Training steps: {len(train_steps)}")
print(f"Evaluation steps: {len(eval_steps)}")
print(f"Evaluation frequency: every {eval_steps[1] - eval_steps[0]} steps\n")

# Find best checkpoint based on multiple criteria
print("="*80)
print("CHECKPOINT ANALYSIS")
print("="*80)

# Best by eval loss (lower is better)
best_loss_idx = np.argmin(eval_loss)
best_loss_step = eval_steps[best_loss_idx]

# Best by F1 (higher is better)
best_f1_idx = np.argmax(eval_cls_f1)
best_f1_step = eval_steps[best_f1_idx]

# Best by R2 (higher is better)
best_r2_idx = np.argmax(eval_reg_r2)
best_r2_step = eval_steps[best_r2_idx]

print(f"\nBest Checkpoints by Metric:")
print(f"{'Metric':<20} {'Best Step':<12} {'Value':<15} {'Index'}")
print("-"*80)
print(f"{'Eval Loss':<20} {best_loss_step:<12} {eval_loss[best_loss_idx]:<15.4f} {best_loss_idx}")
print(f"{'Classification F1':<20} {best_f1_step:<12} {eval_cls_f1[best_f1_idx]:<15.4f} {best_f1_idx}")
print(f"{'Regression R²':<20} {best_r2_step:<12} {eval_reg_r2[best_r2_idx]:<15.4f} {best_r2_idx}")

# Composite score: normalized combination of metrics
# Normalize each metric to [0, 1] scale
norm_loss = 1 - (np.array(eval_loss) - min(eval_loss)) / (max(eval_loss) - min(eval_loss))  # invert: lower is better
norm_f1 = (np.array(eval_cls_f1) - min(eval_cls_f1)) / (max(eval_cls_f1) - min(eval_cls_f1))  # higher is better
norm_r2 = (np.array(eval_reg_r2) - min(eval_reg_r2)) / (max(eval_reg_r2) - min(eval_reg_r2))  # higher is better

# Composite: equal weight to all metrics
composite_score = (norm_loss + norm_f1 + norm_r2) / 3.0
best_composite_idx = np.argmax(composite_score)
best_composite_step = eval_steps[best_composite_idx]

print(f"{'Composite Score':<20} {best_composite_step:<12} {composite_score[best_composite_idx]:<15.4f} {best_composite_idx}")

# Show metrics for user's suggested checkpoint (step 600)
if 600 in eval_steps:
    step_600_idx = eval_steps.index(600)
    print(f"\n* User's Suggested Checkpoint (Step 600):")
    print(f"   Eval Loss:  {eval_loss[step_600_idx]:.4f}")
    print(f"   F1 Score:   {eval_cls_f1[step_600_idx]:.4f}")
    print(f"   R² Score:   {eval_reg_r2[step_600_idx]:.4f}")
    print(f"   Composite:  {composite_score[step_600_idx]:.4f}")
else:
    print(f"\nWARNING: Step 600 not found in evaluation steps")
    print(f"   Available eval steps: {eval_steps[:10]}...")

# Detailed metrics table for early checkpoints (first 10 evals)
print(f"\n\nTABLE: Detailed Metrics (First 10 Evaluations):")
print(f"{'Step':<8} {'Loss':<10} {'Acc':<10} {'F1':<10} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'Composite':<10}")
print("-"*88)
for i in range(min(10, len(eval_steps))):
    print(f"{eval_steps[i]:<8} {eval_loss[i]:<10.4f} {eval_cls_acc[i]:<10.4f} "
          f"{eval_cls_f1[i]:<10.4f} {eval_reg_mae[i]:<10.4f} {eval_reg_rmse[i]:<10.4f} "
          f"{eval_reg_r2[i]:<10.4f} {composite_score[i]:<10.4f}")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Mistral Fine-Tuning Learning Curves', fontsize=16, fontweight='bold')

# 1. Training Loss
ax = axes[0, 0]
ax.plot(train_steps, train_loss, 'b-', alpha=0.3, label='Train Loss')
ax.plot(eval_steps, eval_loss, 'r-', linewidth=2, label='Eval Loss', marker='o')
ax.axvline(best_loss_step, color='g', linestyle='--', alpha=0.5, label=f'Best (step {best_loss_step})')
if 600 in eval_steps:
    ax.axvline(600, color='purple', linestyle='--', alpha=0.5, label='User suggestion (600)')
ax.set_xlabel('Steps')
ax.set_ylabel('Loss')
ax.set_title('Training & Evaluation Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Classification Accuracy
ax = axes[0, 1]
ax.plot(eval_steps, eval_cls_acc, 'b-', linewidth=2, marker='o')
ax.axvline(best_f1_step, color='g', linestyle='--', alpha=0.5, label=f'Best F1 (step {best_f1_step})')
if 600 in eval_steps:
    ax.axvline(600, color='purple', linestyle='--', alpha=0.5, label='User suggestion (600)')
ax.set_xlabel('Steps')
ax.set_ylabel('Accuracy')
ax.set_title('Classification Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Classification F1
ax = axes[0, 2]
ax.plot(eval_steps, eval_cls_f1, 'g-', linewidth=2, marker='o')
ax.axvline(best_f1_step, color='r', linestyle='--', alpha=0.5, label=f'Best (step {best_f1_step})')
if 600 in eval_steps:
    ax.axvline(600, color='purple', linestyle='--', alpha=0.5, label='User suggestion (600)')
ax.set_xlabel('Steps')
ax.set_ylabel('F1 Score')
ax.set_title('Classification F1 Score')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Regression MAE
ax = axes[1, 0]
ax.plot(eval_steps, eval_reg_mae, 'm-', linewidth=2, marker='o')
ax.set_xlabel('Steps')
ax.set_ylabel('MAE')
ax.set_title('Regression MAE (lower is better)')
ax.grid(True, alpha=0.3)

# 5. Regression RMSE
ax = axes[1, 1]
ax.plot(eval_steps, eval_reg_rmse, 'c-', linewidth=2, marker='o')
ax.set_xlabel('Steps')
ax.set_ylabel('RMSE')
ax.set_title('Regression RMSE (lower is better)')
ax.grid(True, alpha=0.3)

# 6. Regression R²
ax = axes[1, 2]
ax.plot(eval_steps, eval_reg_r2, 'orange', linewidth=2, marker='o')
ax.axvline(best_r2_step, color='r', linestyle='--', alpha=0.5, label=f'Best (step {best_r2_step})')
if 600 in eval_steps:
    ax.axvline(600, color='purple', linestyle='--', alpha=0.5, label='User suggestion (600)')
ax.set_xlabel('Steps')
ax.set_ylabel('R² Score')
ax.set_title('Regression R² Score')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save plot
output_path = checkpoint_dir.parent / "mistral_learning_curves.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n\nOK: Learning curves saved to:\n  {output_path}")

# Create a zoomed-in plot for early training (first 2000 steps)
early_eval_indices = [i for i, s in enumerate(eval_steps) if s <= 2000]
if len(early_eval_indices) > 0:
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle('Early Training Analysis (First 2000 Steps)', fontsize=14, fontweight='bold')

    # Focus on key metrics
    ax = axes2[0]
    ax.plot([eval_steps[i] for i in early_eval_indices],
            [eval_loss[i] for i in early_eval_indices],
            'r-', linewidth=2, marker='o', markersize=8)
    if 600 in eval_steps and 600 <= 2000:
        idx_600 = eval_steps.index(600)
        ax.axvline(600, color='purple', linestyle='--', alpha=0.7, linewidth=2)
        ax.plot(600, eval_loss[idx_600], 'purple', marker='*', markersize=20, label='Step 600')
    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel('Eval Loss', fontsize=12)
    ax.set_title('Evaluation Loss (Early Training)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    ax.plot([eval_steps[i] for i in early_eval_indices],
            [eval_cls_f1[i] for i in early_eval_indices],
            'g-', linewidth=2, marker='o', markersize=8)
    if 600 in eval_steps and 600 <= 2000:
        ax.axvline(600, color='purple', linestyle='--', alpha=0.7, linewidth=2)
        ax.plot(600, eval_cls_f1[idx_600], 'purple', marker='*', markersize=20, label='Step 600')
    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Classification F1 (Early Training)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes2[2]
    ax.plot([eval_steps[i] for i in early_eval_indices],
            [eval_reg_r2[i] for i in early_eval_indices],
            'orange', linewidth=2, marker='o', markersize=8)
    if 600 in eval_steps and 600 <= 2000:
        ax.axvline(600, color='purple', linestyle='--', alpha=0.7, linewidth=2)
        ax.plot(600, eval_reg_r2[idx_600], 'purple', marker='*', markersize=20, label='Step 600')
    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Regression R² (Early Training)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path_early = checkpoint_dir.parent / "mistral_learning_curves_early.png"
    plt.savefig(output_path_early, dpi=300, bbox_inches='tight')
    print(f"OK: Early training curves saved to:\n  {output_path_early}")

plt.show()

# Final recommendation
print("\n" + "="*80)
print("* RECOMMENDATION")
print("="*80)

if 600 in eval_steps:
    step_600_idx = eval_steps.index(600)

    # Compare step 600 to best checkpoints
    is_best_loss = best_loss_step == 600
    is_best_f1 = best_f1_step == 600
    is_best_r2 = best_r2_step == 600

    print(f"\nCheckpoint at Step 600:")
    print(f"  Eval Loss:     {eval_loss[step_600_idx]:.4f} {'OK BEST' if is_best_loss else f'(best: {eval_loss[best_loss_idx]:.4f} at step {best_loss_step})'}")
    print(f"  F1 Score:      {eval_cls_f1[step_600_idx]:.4f} {'OK BEST' if is_best_f1 else f'(best: {eval_cls_f1[best_f1_idx]:.4f} at step {best_f1_step})'}")
    print(f"  R² Score:      {eval_reg_r2[step_600_idx]:.4f} {'OK BEST' if is_best_r2 else f'(best: {eval_reg_r2[best_r2_idx]:.4f} at step {best_r2_step})'}")
    print(f"  Composite:     {composite_score[step_600_idx]:.4f} (best: {composite_score[best_composite_idx]:.4f} at step {best_composite_step})")

    # Rank step 600
    loss_rank = sorted(range(len(eval_loss)), key=lambda i: eval_loss[i]).index(step_600_idx) + 1
    f1_rank = sorted(range(len(eval_cls_f1)), key=lambda i: -eval_cls_f1[i]).index(step_600_idx) + 1
    r2_rank = sorted(range(len(eval_reg_r2)), key=lambda i: -eval_reg_r2[i]).index(step_600_idx) + 1

    print(f"\n  Rankings (out of {len(eval_steps)} checkpoints):")
    print(f"    Loss:  #{loss_rank}")
    print(f"    F1:    #{f1_rank}")
    print(f"    R²:    #{r2_rank}")

    avg_rank = (loss_rank + f1_rank + r2_rank) / 3
    print(f"    Average rank: #{avg_rank:.1f}")

    if avg_rank <= 3:
        print(f"\n  OK: Step 600 is an EXCELLENT choice (top-3 average ranking)!")
    elif avg_rank <= 5:
        print(f"\n  OK Step 600 is a GOOD choice (top-5 average ranking)")
    else:
        print(f"\n  WARNING: Step 600 is reasonable but not optimal")
        print(f"     Consider checkpoint at step {best_composite_step} (best composite score)")
else:
    print("\nWARNING: Step 600 checkpoint not found in evaluation history")

print("\n" + "="*80)
print("OK: Analysis complete!")
print("="*80)
