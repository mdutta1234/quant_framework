# experiments/train_rl.py
import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from pipeline.runner import TradingPipeline
from utils.rl_helper import generate_predictor_outputs
from modules.environment.trading_env import MultiAssetTradingEnv

def run_rl_pipeline():
    # 1. Run Core Pipeline to get trained predictors and scaled datasets
    config_path = os.path.join(ROOT_DIR, "configs", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    pipeline = TradingPipeline(config)
    feature_datasets = pipeline.run()
    
    # 2. Synthesize Predictions
    predictions_dict = generate_predictor_outputs(
        models=pipeline.predictors,
        feature_datasets=feature_datasets,
        scaler=pipeline.scaler,
        config=config
    )
    
    # 3. Instantiate Gymnasium Environment for TEST split
    test_df = feature_datasets['test']
    test_preds = predictions_dict['test']
    
    env = MultiAssetTradingEnv(df=test_df, predictions_df=test_preds, config=config)
    
    # 4. Execute Benchmark Strategies on Test Split
    print("\n--- STAGE 17.2: Running Backtest Strategies ---")
    
    # Strategy A: Equal Weight (Buy & Hold)
    eq_returns = run_equal_weight_strategy(env)
    
    # Strategy B: Predictor-Only Top Pick
    pred_returns = run_predictor_only_strategy(env)
    
    # 5. Plot Comparison
    results_dir = os.path.join(ROOT_DIR, "experiments", "plots")
    os.makedirs(results_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(eq_returns, label="Equal-Weight Baseline", color="black", linestyle="--")
    plt.plot(pred_returns, label="Predictor-Only Strategy", color="blue", linewidth=2)
    plt.title("Stage 17 Portfolio Value Backtest (Test Set)")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "RL_Environment_Backtest.png"))
    plt.close()
    
    print(f"[*] Trading Environment execution complete. Plot saved to {results_dir}/RL_Environment_Backtest.png")

def run_equal_weight_strategy(env):
    obs, info = env.reset()
    done = False
    num_assets = env.num_assets
    action = np.ones(num_assets, dtype=np.float32) / num_assets
    
    values = [env.initial_cash]
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        values.append(info['portfolio_value'])
        done = terminated or truncated
    return values

def run_predictor_only_strategy(env):
    obs, info = env.reset()
    done = False
    num_assets = env.num_assets
    
    values = [env.initial_cash]
    while not done:
        # Extract predicted returns from state observation
        # per-stock block size = len(market_features) + 4
        block_size = len(env.market_feature_cols) + 4
        pred_returns = [obs[i * block_size + len(env.market_feature_cols)] for i in range(num_assets)]
        
        # Allocate proportionally to positive predicted returns
        pos_preds = np.maximum(0.0, pred_returns)
        p_sum = np.sum(pos_preds)
        
        if p_sum > 0:
            action = pos_preds / p_sum
        else:
            action = np.zeros(num_assets, dtype=np.float32)
            
        obs, reward, terminated, truncated, info = env.step(action)
        values.append(info['portfolio_value'])
        done = terminated or truncated
    return values

if __name__ == "__main__":
    run_rl_pipeline()