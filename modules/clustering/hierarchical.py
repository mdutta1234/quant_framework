# modules/clustering/hierarchical.py
import pandas as pd
import numpy as np
from typing import Dict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from core.base_clusterer import BaseClusterer

class HierarchicalClusterer(BaseClusterer):
    EPSILON = 1e-8
    
    def fit_predict(self, train_features: pd.DataFrame) -> Dict[str, int]:
        print("\n[CLUSTERING] Summarizing stock dynamics...")
        tickers = train_features['ticker'].unique()
        
        # 1. Compute Descriptors
        descriptor_matrix = []
        for ticker in tickers:
            ticker_df = train_features[train_features['ticker'] == ticker].copy()
            
            ticker_desc = {'ticker': ticker}
            for desc in self.descriptors_list:
                method_name = f"_calc_{desc}"
                if hasattr(self, method_name):
                    method = getattr(self, method_name)
                    val = method(ticker_df)
                    # Convert single NaN / Inf descriptor values to 0.0 immediately
                    ticker_desc[desc] = 0.0 if (pd.isna(val) or np.isinf(val)) else val
                else:
                    raise NotImplementedError(f"Descriptor '{desc}' not implemented.")
            
            ticker_desc['_raw_volume'] = ticker_df['volume'].mean()
            descriptor_matrix.append(ticker_desc)
            
        desc_df = pd.DataFrame(descriptor_matrix).set_index('ticker')
        
        # --- THE SUPER SCRUB ---
        # Force all columns to float, convert any sneaky Infs to NaNs, then fill NaNs with 0.0
        desc_df = desc_df.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        # 2. Normalize descriptors safely (Z-score)
        clustering_features = desc_df.drop(columns=['_raw_volume'])
        std_devs = clustering_features.std().replace(0, self.EPSILON)
        
        # Scrub standard deviations just in case Pandas returned NaN for constant columns
        std_devs = std_devs.fillna(1.0) 
        
        clustering_features = (clustering_features - clustering_features.mean()) / std_devs
        
        # Final scrub on the normalized data
        clustering_features = clustering_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 3. Compute Distance & Linkage
        metric = self.config.get('distance_metric', 'correlation')
        method = self.config.get('linkage_method', 'average')
        
        print(f"[CLUSTERING] Building Hierarchical Tree (Metric: {metric})...")
        
        # Compute pairwise distance
        dist_matrix = pdist(clustering_features, metric=metric)
        
        # --- THE SCI-PY FAILSAFE ---
        # Correlation distance maxes out at 2.0. If pdist returned NaN (e.g., zero variance row),
        # we force it to 2.0 so the linkage algorithm safely ignores them instead of crashing.
        dist_matrix = np.nan_to_num(dist_matrix, nan=2.0, posinf=2.0, neginf=2.0)
        
        Z = linkage(dist_matrix, method=method)
        
        # 4. Harvest Clusters & Apply Size Constraints
        min_size = self.config['constraints']['min_size']
        max_size = self.config['constraints']['max_size']
        
        candidate_clusters = {}
        for n_clust in range(2, max(3, len(tickers) // 2)):
            labels = fcluster(Z, n_clust, criterion='maxclust')
            for c_id in np.unique(labels):
                cluster_tickers = desc_df.index[labels == c_id].tolist()
                if min_size <= len(cluster_tickers) <= max_size:
                    c_set = frozenset(cluster_tickers)
                    if c_set not in candidate_clusters:
                        candidate_clusters[c_set] = cluster_tickers
                        
        print(f"[CLUSTERING] Found {len(candidate_clusters)} candidate clusters of size {min_size}-{max_size}.")

        # 5. Rank Clusters
        cluster_scores = []
        for i, c_tickers in enumerate(candidate_clusters.values()):
            sub_df = clustering_features.loc[c_tickers]
            if len(c_tickers) > 1:
                internal_dist = pdist(sub_df, metric=metric)
                # Clean dist before taking mean
                internal_dist = np.nan_to_num(internal_dist, nan=2.0)
                internal_similarity = 1.0 / (internal_dist.mean() + self.EPSILON)
            else:
                internal_similarity = 0
                
            avg_liquidity = desc_df.loc[c_tickers, '_raw_volume'].mean()
            score = internal_similarity * np.log1p(max(0, avg_liquidity))
            cluster_scores.append((score, c_tickers))
            
        cluster_scores.sort(key=lambda x: x[0], reverse=True)
        
        # 6. Select Top Clusters (Mutually Exclusive)
        target_clusters = self.config['constraints']['target_clusters']
        final_mapping = {}
        assigned_tickers = set()
        cluster_id_counter = 0
        
        for score, c_tickers in cluster_scores:
            if cluster_id_counter >= target_clusters:
                break
            if any(t in assigned_tickers for t in c_tickers):
                continue
                
            for t in c_tickers:
                final_mapping[t] = cluster_id_counter
                assigned_tickers.add(t)
                
            print(f"  -> Selected Cluster {cluster_id_counter} | Score: {score:.2f} | Tickers: {c_tickers}")
            cluster_id_counter += 1

        return final_mapping

    # ==========================================
    # DESCRIPTOR FUNCTIONS (All 30 implemented)
    # ==========================================
    
    # --- A. Return Distribution ---
    def _calc_mean_daily_return(self, df): return df['close'].pct_change().mean()
    def _calc_median_daily_return(self, df): return df['close'].pct_change().median()
    def _calc_return_std(self, df): return df['close'].pct_change().std()
    def _calc_return_var(self, df): return df['close'].pct_change().var()
    def _calc_return_skewness(self, df): return df['close'].pct_change().skew()
    def _calc_return_kurtosis(self, df): return df['close'].pct_change().kurt()
    def _calc_max_daily_return(self, df): return df['close'].pct_change().max()
    def _calc_min_daily_return(self, df): return df['close'].pct_change().min()

    # --- B. Trend Characteristics ---
    def _calc_mean_sma20_dist(self, df):
        sma20 = df['close'].rolling(20).mean()
        return ((df['close'] - sma20) / sma20).mean()
        
    def _calc_mean_ema20_dist(self, df):
        ema20 = df['close'].ewm(span=20, adjust=False).mean()
        return ((df['close'] - ema20) / ema20).mean()
        
    def _calc_pct_days_above_sma20(self, df):
        sma20 = df['close'].rolling(20).mean()
        return (df['close'] > sma20).mean()
        
    def _calc_pct_days_above_sma50(self, df):
        sma50 = df['close'].rolling(50).mean()
        return (df['close'] > sma50).mean()
        
    def _calc_mean_momentum_20(self, df):
        return (df['close'] - df['close'].shift(20)).mean()

    # --- C. Volatility Profile ---
    def _calc_historical_volatility(self, df):
        # Annualized volatility (assuming ~252 trading days)
        return df['close'].pct_change().std() * np.sqrt(252)
        
    def _calc_rolling_vol_mean(self, df):
        return df['close'].pct_change().rolling(20).std().mean()
        
    def _calc_rolling_vol_std(self, df):
        return df['close'].pct_change().rolling(20).std().std()
        
    def _calc_atr_mean(self, df):
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.mean()
        
    def _calc_bb_width_mean(self, df):
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        return ((4 * std20) / sma20).mean()

    # --- D. Volume Behaviour ---
    def _calc_mean_volume(self, df): return df['volume'].mean()
    def _calc_volume_std(self, df): return df['volume'].std()
    def _calc_mean_volume_change(self, df): return df['volume'].pct_change().mean()
    def _calc_volume_autocorr(self, df): return df['volume'].autocorr(lag=1)

    # --- E. Price Dynamics ---
    def _calc_mean_high_low_pct(self, df):
        return ((df['high'] - df['low']) / df['close']).mean()
        
    def _calc_mean_open_close_pct(self, df):
        return ((df['close'] - df['open']) / df['open']).mean()
        
    def _calc_mean_overnight_gap_pct(self, df):
        prev_close = df['close'].shift(1)
        return ((df['open'] - prev_close) / prev_close).mean()
        
    def _calc_price_autocorr_lag1(self, df): return df['close'].autocorr(lag=1)
    def _calc_price_autocorr_lag5(self, df): return df['close'].autocorr(lag=5)

    # --- F. Candlestick Behaviour ---
    def _calc_mean_candle_body_ratio(self, df):
        body = (df['close'] - df['open']).abs()
        rng = df['high'] - df['low']
        # Avoid division by zero
        return (body / (rng + 1e-8)).mean()
        
    def _calc_mean_upper_shadow_ratio(self, df):
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        rng = df['high'] - df['low']
        return (upper_shadow / (rng + 1e-8)).mean()
        
    def _calc_mean_lower_shadow_ratio(self, df):
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        rng = df['high'] - df['low']
        return (lower_shadow / (rng + 1e-8)).mean()