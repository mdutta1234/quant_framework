import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from core.factory import ModuleFactory
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.rl_helper import generate_predictor_outputs            
from modules.environment.trading_env import MultiAssetTradingEnv


class TradingPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.universe_selector = ModuleFactory.get_universe(config['universe_selector'])
        self.cleaner = ModuleFactory.get_cleaner(config['data_cleaner'])
        self.feature_eng = ModuleFactory.get_feature_engineer(config['feature_engineer'])
        self.clusterer = ModuleFactory.get_clusterer(config['cluster_selector'])

    def run(self):
        print("\n--- STAGE 1: Universe Selection & Data Loading ---")
        raw_datasets = self.universe_selector.fetch_data()
        
        print("\n--- STAGE 2: Data Cleaning & Alignment ---")
        cleaned_datasets = self.cleaner.clean(raw_datasets)
        
        print("\n--- STAGE 3: Feature Engineering ---")
        feature_datasets, input_dim = self.feature_eng.process(cleaned_datasets)
        
        print(f"\n[FEATURE CHECK]")
        print(f"Total features extracted (Dynamic Input Dimension): {input_dim}")
        print("\nSample of final feature matrix (Train):")
        print(feature_datasets['train'].head(3))
        
        print("\n--- STAGE 4: Stock Similarity & Clustering ---")
        # Pass ONLY the train split to prevent look-ahead bias
        cluster_mapping = self.clusterer.fit_predict(feature_datasets['train'])
        
        # Filter all datasets to ONLY include stocks that made it into a cluster
        selected_tickers = list(cluster_mapping.keys())
        local_id_mapping = {}
        for c_id in set(cluster_mapping.values()):
            c_tickers = [t for t, cid in cluster_mapping.items() if cid == c_id]
            for local_idx, t in enumerate(c_tickers):
                local_id_mapping[t] = local_idx

        for split_name in feature_datasets.keys():
            df = feature_datasets[split_name]
            df = df[df['ticker'].isin(selected_tickers)].copy()
            
            # Inject the cluster_id AND local_id into the dataframe
            df['cluster_id'] = df['ticker'].map(cluster_mapping)
            df['local_id'] = df['ticker'].map(local_id_mapping) # <--- INJECTED HERE
            feature_datasets[split_name] = df
            
        num_clusters = len(set(cluster_mapping.values()))
        print(f"\n[*] Universe reduced to {len(selected_tickers)} highly correlated/liquid stocks across {num_clusters} clusters.")
        # ==========================================
        # STAGE 5: Mathematical Consensus Prior
        # ==========================================
        print("\n--- STAGE 5: Mathematical Priors ---")
        
        # Initialize Prior Model
        self.prior = ModuleFactory.get_prior(self.config['prior_model'])
        
        # Fit on train data (if the prior supports MLE fitting, otherwise it passes)
        self.prior.fit(feature_datasets['train'])
        
        # Generate expected future feature vectors dynamically
        expected_datasets = self.prior.generate_prior(feature_datasets)
        
        print(f"\n[PRIOR CHECK]")
        print("Actual features (Train) Row 100:")
        print(feature_datasets['train'].iloc[100][['open', 'close', 'rsi_14', 'macd']])
        
        print("\nExpected features (Consensus Prior) Row 100:")
        print(expected_datasets['train'].iloc[100][['open', 'close', 'rsi_14', 'macd']])

        # Phase 2 Dynamic Config Injection setup
        # ==========================================
        # STAGE 6: Confidence Weight Generation
        # ==========================================
        print("\n--- STAGE 6: Confidence Weight Generation ---")
        self.weight_gen = ModuleFactory.get_weight_generator(self.config['weight_generator'])
        weight_datasets = self.weight_gen.generate_weights(feature_datasets, expected_datasets)
        
        # Merge weights into the final feature datasets so they are saved together
        for split_name in feature_datasets.keys():
            feature_datasets[split_name] = pd.merge(
                feature_datasets[split_name], 
                weight_datasets[split_name], 
                on=['ticker', 'date'], 
                how='left'
            )

        # Phase 2 Dynamic Config Injection setup
        self.config['predictor'] = self.config.get('predictor', {})
        self.config['predictor']['input_dim'] = input_dim
        self.config['predictor']['num_clusters'] = num_clusters

        # ==========================================
        # STAGE 6.5: Save Processed & Clustered Data to CSV
        # ==========================================
        pipeline_settings = self.config.get('pipeline_settings', {})
        if pipeline_settings.get('save_features_to_csv', False):
            save_dir = pipeline_settings.get('processed_data_dir', './data/processed')
            os.makedirs(save_dir, exist_ok=True)
            print(f"\n--- STAGE 4.5: Saving Final Feature Matrices (with Cluster IDs) to Disk ---")
            
            for split_name, df in feature_datasets.items():
                for ticker, ticker_df in df.groupby('ticker'):
                    file_name = f"{split_name}_{ticker}.csv"
                    file_path = os.path.join(save_dir, file_name)
                    ticker_df.to_csv(file_path, index=False)
                    
            print(f"[*] Successfully saved all processed features to '{save_dir}/'")

        # ==========================================
        # STAGE 7: Dynamic Neural Network Initialization
        # ==========================================
        print("\n--- STAGE 7: Prediction Module (Neural Networks) ---")
        
        self.predictors = {} # Dictionary to hold 1 model per cluster
        
        for c_id in range(num_clusters):
            # Find how many stocks are in this specific cluster
            cluster_stocks = [t for t, cid in cluster_mapping.items() if cid == c_id]
            num_stocks_in_cluster = len(cluster_stocks)
            
            # Dynamically inject dimensions into a localized config
            pred_config = self.config['predictor'].copy()
            pred_config['input_dim'] = input_dim
            pred_config['output_dim'] = input_dim
            pred_config['id_dim'] = num_stocks_in_cluster # <--- DYNAMIC ONE-HOT DIMENSION!
            
            # Instantiate the Deep Learning model for this cluster
            model = ModuleFactory.get_predictor(pred_config)
            self.predictors[c_id] = model
            
            print(f"  -> Initialized {pred_config['type']} for Cluster {c_id}")
            print(f"     Stocks: {num_stocks_in_cluster} | Input Dim: {input_dim} + {num_stocks_in_cluster} (ID) | Output: {input_dim}")
            
            # Quick PyTorch Tensor test to prove the architecture works
            import torch
            dummy_x = torch.randn(32, pred_config['seq_len'], input_dim) # Batch=32
            dummy_id = torch.randint(0, num_stocks_in_cluster, (32,))    # Random stock IDs
            
            with torch.no_grad():
                dummy_out = model(dummy_x, dummy_id)
            assert dummy_out.shape == (32, input_dim), "Model output shape mismatch!"
            
        print("\n[*] All cluster-specific Neural Networks successfully built and shape-tested!")
        
        # ==========================================
        # STAGE 14: Loss Function Initialization
        # ==========================================
        print("\n--- STAGE 8: Custom Weighted Loss Function ---")
        
        self.loss_fn = ModuleFactory.get_loss(self.config['loss_function'])
        print(f"  -> Initialized: {self.config['loss_function']['type']}")
        
        # --- PYTORCH LOSS TEST ---
        import torch
        
        dummy_preds = torch.tensor([[1.5, -0.5], [2.0, 1.0]], dtype=torch.float32)
        dummy_targets = torch.tensor([[1.0, 0.0], [2.0, 1.0]], dtype=torch.float32)
        
        # Row 1 has high confidence (1.0), Row 2 has low confidence (0.1)
        dummy_weights = torch.tensor([1.0, 0.1], dtype=torch.float32) 
        
        loss_val = self.loss_fn(dummy_preds, dummy_targets, dummy_weights)
        print(f"  -> Test Loss Evaluation Successful! Computed Loss: {loss_val.item():.4f}")
        
        # ==========================================
        # STAGE 14.5: Feature Scaling
        # ==========================================
        
        print("\n--- STAGE 14.5: Feature Scaling (Z-Score Normalization) ---")
        from sklearn.preprocessing import StandardScaler
        
        feature_cols = self.feature_eng.features_list
        self.scaler = StandardScaler()
        
        # ---> THE FIX: Cast all features to float to prevent integer override errors <---
        for split_name in feature_datasets.keys():
            feature_datasets[split_name][feature_cols] = feature_datasets[split_name][feature_cols].astype(float)
        
        # 1. Fit ONLY on the training split to prevent look-ahead bias
        self.scaler.fit(feature_datasets['train'][feature_cols])
        
        # 2. Transform Train, Val, and Test
        for split_name, df in feature_datasets.items():
            feature_datasets[split_name].loc[:, feature_cols] = self.scaler.transform(df[feature_cols])
            
        print("  -> Features successfully scaled to Mean=0, Std=1")
        # ==========================================
        # STAGE 15: PyTorch DataLoaders
        # ==========================================
        print("\n--- STAGE 15: Building PyTorch DataLoaders ---")
        from utils.dataset import QuantTimeSeriesDataset
        from torch.utils.data import DataLoader
        
        # We need the feature column names to feed the dataset (exclude metadata/weights)
        feature_cols = self.feature_eng.features_list
        print(len(feature_cols))
        print(feature_cols)
        seq_len = self.config['predictor']['seq_len']
        batch_size = 64
        
        self.dataloaders = {}
        
        for split_name, df in feature_datasets.items():
            print(f"\nProcessing {split_name.upper()} split...")
            
            # Create Dataset
            dataset = QuantTimeSeriesDataset(df, seq_len=seq_len, feature_cols=feature_cols)
            
            # Create DataLoader (Shuffle only the training set)
            shuffle = True if split_name == 'train' else False
            
            self.dataloaders[split_name] = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle,
                num_workers=0, # Set to >0 if running on a heavy machine
                drop_last=True
            )
            
        # --- PYTORCH BATCH TEST ---
        print("\n[DATALOADER CHECK]")
        # Grab exactly 1 batch of training data
        batch_x, batch_id, batch_cid, batch_y, batch_w = next(iter(self.dataloaders['train']))
        
        print(f"Historical X shape : {batch_x.shape} (batch, seq_len, features)")
        print(f"Local ID shape     : {batch_id.shape} (batch)")
        print(f"Cluster ID shape   : {batch_cid.shape} (batch)")
        print(f"Target Y shape     : {batch_y.shape} (batch, features)")
        print(f"Weight W shape     : {batch_w.shape} (batch)")
        
        print("\n[*] Data perfectly formatted for the PyTorch Training Loop!")

        # ==========================================
        # STAGE 16: Multi-Model Training
        # ==========================================
        from utils.trainer import MultiClusterTrainer
        
        trainer = MultiClusterTrainer(
            models=self.predictors,
            dataloaders=self.dataloaders,
            loss_fn=self.loss_fn,
            config=self.config
        )
        
        trainer.train()
        
        # ==========================================
        # STAGE 17: Inference Plotting & Export
        # ==========================================
        print("\n--- STAGE 17: Inference Plotting (Actual vs Predicted) ---")

        output_dir = "./main_py_results"
        os.makedirs(output_dir, exist_ok=True)

        feature_cols = self.feature_eng.features_list
        ohclv_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in feature_cols]
        ohclv_indices = [feature_cols.index(c) for c in ohclv_cols]

        # Select 1 representative stock from each cluster
        sample_tickers = []
        for c_id in range(num_clusters):
            c_tickers = [t for t, cid in cluster_mapping.items() if cid == c_id]
            if c_tickers:
                sample_tickers.append(c_tickers[0])

        test_df = feature_datasets['test']

        for ticker in sample_tickers:
            t_data = test_df[test_df['ticker'] == ticker]
            if len(t_data) <= seq_len:
                continue

            features = t_data[feature_cols].values.astype(np.float32)
            local_id = t_data['local_id'].values[0]
            cluster_id = t_data['cluster_id'].values[0]

            model = self.predictors[cluster_id]
            model.eval()

            y_true_list, y_pred_list = [], []

            with torch.no_grad():
                for i in range(len(features) - seq_len):
                    x_win = torch.tensor(features[i : i + seq_len]).unsqueeze(0).to(trainer.device)
                    id_tens = torch.tensor([local_id]).to(trainer.device)

                    pred = model(x_win, id_tens).cpu().numpy()[0]
                    y_pred_list.append(pred)
                    y_true_list.append(features[i + seq_len])

            y_pred_scaled = np.array(y_pred_list)
            y_true_scaled = np.array(y_true_list)

            # Reverse Z-score normalization
            y_pred_real = self.scaler.inverse_transform(y_pred_scaled)
            y_true_real = self.scaler.inverse_transform(y_true_scaled)

            # Generate individual OHCLV plots for this stock
            stock_dir = os.path.join(output_dir, f"cluster_{cluster_id}_{ticker}")
            os.makedirs(stock_dir, exist_ok=True)

            for col_name, idx in zip(ohclv_cols, ohclv_indices):
                plt.figure(figsize=(12, 6))
                plt.plot(y_true_real[:, idx], label='Actual', color='black', alpha=0.8)
                plt.plot(y_pred_real[:, idx], label='Predicted', color='crimson', alpha=0.7, linestyle='--')
                plt.title(f"Test Split Inference: {ticker} (Cluster {cluster_id}) - {col_name.upper()}")
                plt.xlabel("Test Sequence Step (Days)")
                plt.ylabel("Real Unscaled Value")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(stock_dir, f"{col_name}.png"))
                plt.close()

            print(f"  -> Saved test inference plots for {ticker} (Cluster {cluster_id}) to {stock_dir}/")

        print(f"\n[*] All pipeline results saved to '{output_dir}/'.")
        # ==========================================
        # STAGE 18: RL Environment Benchmarks
        # ==========================================
        print("\n--- STAGE 18: RL Environment Benchmark Strategies ---")
        
        # 1. Synthesize Predictions for the RL Env
        predictions_dict = generate_predictor_outputs(
            models=self.predictors,
            feature_datasets=feature_datasets,
            scaler=self.scaler,
            config=self.config
        )
        
        test_df = feature_datasets['test']
        test_preds = predictions_dict['test']
        
        # 2. Instantiate Gymnasium Environment
        env = MultiAssetTradingEnv(df=test_df, predictions_df=test_preds, config=self.config)
        
        # 3. Execute Benchmark Strategies
        eq_returns = self._run_equal_weight_strategy(env)
        pred_returns = self._run_predictor_only_strategy(env)
        
        # 4. Plot Comparison
        plt.figure(figsize=(12, 6))
        plt.plot(eq_returns, label="Equal-Weight Baseline", color="black", linestyle="--")
        plt.plot(pred_returns, label="Predictor-Only Strategy", color="blue", linewidth=2)
        plt.title("Stage 18: Portfolio Value Backtest (Test Set)")
        plt.xlabel("Trading Days")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        rl_plot_path = os.path.join(output_dir, "RL_Environment_Backtest.png")
        plt.savefig(rl_plot_path)
        plt.close()
        
        print(f"  -> Saved RL backtest plot to {rl_plot_path}")
        print(f"\n[*] ALL STAGES COMPLETE. Pipeline results are in '{output_dir}/'.")

        return feature_datasets
    
    # ---------------------------------------------------------
    # RL BENCHMARK HELPER METHODS
    # ---------------------------------------------------------
    def _run_equal_weight_strategy(self, env):
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

    def _run_predictor_only_strategy(self, env):
        obs, info = env.reset()
        done = False
        num_assets = env.num_assets
        
        values = [env.initial_cash]
        while not done:
            # Extract predicted returns from the state observation.
            # Using the exact dimensional sizes calculated by the environment
            pred_returns = []
            for i in range(num_assets):
                idx = i * env.per_stock_dim + len(env.market_feature_cols)
                pred_returns.append(obs[idx])
            
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

# Prepare a document in which we repeat this experiment over a number of stocks, chosen randomly , a number of times, and we have a table , where rows tell us the 
# the metric averaged (averaged over all stocks for a give slot, and in all the number of random stock selection) and column tell us the alpha values
# Have this document for various predictor models, and mathmatical priors, and descriptors and cluster selector and over different years of train,test,validate, and over different stock universes etc