import os
import pandas as pd
from core.factory import ModuleFactory

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
        for split_name in feature_datasets.keys():
            df = feature_datasets[split_name]
            df = df[df['ticker'].isin(selected_tickers)].copy()
            
            # Inject the cluster_id into the dataframe
            df['cluster_id'] = df['ticker'].map(cluster_mapping)
            feature_datasets[split_name] = df
            
        num_clusters = len(set(cluster_mapping.values()))
        print(f"\n[*] Universe reduced to {len(selected_tickers)} highly correlated/liquid stocks across {num_clusters} clusters.")
        
        # Phase 2 Dynamic Config Injection setup
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
        
        return feature_datasets