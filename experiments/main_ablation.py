# experiments/main_ablation.py
import sys
import os
import yaml
import matplotlib.pyplot as plt
import pandas as pd

# Add the root directory to Python path so it can import framework modules
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from pipeline.runner import TradingPipeline
from utils.evaluator import QuantEvaluator

def load_config_direct():
    config_path = os.path.join(ROOT_DIR, "configs", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_ablation():
    config = load_config_direct()
    
    # Target directory inside experiments/
    exp_plots_dir = os.path.join(ROOT_DIR, "experiments", "plots")
    os.makedirs(exp_plots_dir, exist_ok=True)
    
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    all_ablation_metrics = pd.DataFrame()
    macro_metrics_list = []
    
    # Set lower epoch count for quick ablation iteration
    config['predictor']['epochs'] = 5 
    
    for alpha in alpha_values:
        print(f"\n========================================")
        print(f" STARTING ABLATION RUN FOR ALPHA = {alpha}")
        print(f"========================================")
        
        config['weight_generator']['alpha'] = alpha
        
        # 1. Run Pipeline
        pipeline = TradingPipeline(config)
        feature_datasets = pipeline.run()
        
        # 2. Run Evaluator
        evaluator = QuantEvaluator(
            models=pipeline.predictors,
            scaler=pipeline.scaler,
            config=config,
            feature_cols=pipeline.feature_eng.features_list
        )
        
        metrics_df = evaluator.evaluate_and_plot(feature_datasets, alpha_val=alpha)
        all_ablation_metrics = pd.concat([all_ablation_metrics, metrics_df], ignore_index=True)
        
        # 3. Macro Metrics
        train_loss = metrics_df[metrics_df['split'] == 'train']['Weighted_RMSE'].mean()
        val_loss = metrics_df[metrics_df['split'] == 'val']['Weighted_RMSE'].mean()
        
        gen_gap = val_loss - train_loss
        overfit_ratio = val_loss / (train_loss + 1e-8)
        
        print(f"\n[METRICS for Alpha {alpha}]")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Generalisation Gap: {gen_gap:.4f}")
        print(f"Overfitting Ratio: {overfit_ratio:.4f} (1.0 = Perfect Balance)")
        
        macro_metrics_list.append({
            'alpha': alpha,
            'Generalisation_Gap': gen_gap,
            'Overfitting_Ratio': overfit_ratio
        })
        
    # 4. Save Plots inside experiments/plots/
    print(f"\n[ABLATION] Saving plots to {exp_plots_dir}...")
    
    test_metrics = all_ablation_metrics[all_ablation_metrics['split'] == 'test']
    alpha_grouped = test_metrics.groupby('alpha').mean(numeric_only=True).reset_index()
    
    standard_metrics = [
        'MAE', 'RMSE', 'sMAPE', 'MASE', 'DirAcc', 
        'DirF1', 'RetCorr', 'IC', 'Weighted_MAE', 'Weighted_RMSE'
    ]
    
    for metric in standard_metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(alpha_grouped['alpha'], alpha_grouped[metric], marker='o', linewidth=2, color='blue')
        plt.title(f"{metric} vs Alpha (Huber Weight Penalty)")
        plt.xlabel("Alpha (0 = Ignore Prior, 1 = Max Penalty)")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(exp_plots_dir, f"Alpha_Ablation_{metric}.png"))
        plt.close()
        
    macro_df = pd.DataFrame(macro_metrics_list)
    
    # Plot Generalisation Gap
    plt.figure(figsize=(10, 6))
    plt.plot(macro_df['alpha'], macro_df['Generalisation_Gap'], marker='s', linewidth=2, color='purple')
    plt.title("Generalisation Gap vs Alpha\n(Validation Loss - Training Loss)")
    plt.xlabel("Alpha")
    plt.ylabel("Gap (Lower is better)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(exp_plots_dir, "Alpha_Ablation_Generalisation_Gap.png"))
    plt.close()

    # Plot Overfitting Ratio
    plt.figure(figsize=(10, 6))
    plt.plot(macro_df['alpha'], macro_df['Overfitting_Ratio'], marker='^', linewidth=2, color='orange')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Perfect Balance (1.0)')
    plt.title("Overfitting Ratio vs Alpha\n(Validation Loss / Training Loss)")
    plt.xlabel("Alpha")
    plt.ylabel("Ratio (Closer to 1.0 is better)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(exp_plots_dir, "Alpha_Ablation_Overfitting_Ratio.png"))
    plt.close()
        
    print(f"[*] Ablation study complete. Results saved in {exp_plots_dir}")

if __name__ == "__main__":
    run_ablation()