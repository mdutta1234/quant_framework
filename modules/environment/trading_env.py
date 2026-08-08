import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class MultiAssetTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, predictions_df: pd.DataFrame, config: dict, scaler=None, feature_cols=None):
        super().__init__()
        
        self.config = config['rl_environment']
        self.initial_cash = self.config.get('initial_cash', 100000.0)
        self.transaction_cost_pct = self.config.get('transaction_cost_pct', 0.0010)
        self.slippage_pct = self.config.get('slippage_pct', 0.0005)
        
        self.max_stock_weight = self.config.get('max_stock_weight', 0.40)
        self.min_stock_weight = self.config.get('min_stock_weight', 0.00)
        self.max_total_exposure = self.config.get('max_total_exposure', 1.00)
        
        self.reward_cfg = self.config.get('reward_config', {})
        
        self.scaler = scaler
        self.feature_cols = feature_cols
        
        self.tickers = sorted(df['ticker'].unique().tolist())
        self.num_assets = len(self.tickers)
        self.dates = sorted(df['date'].unique().tolist())
        self.total_steps = len(self.dates)
        
        exclude_cols = ['ticker', 'date', 'cluster_id', 'local_id', 'weight']
        self.market_feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        self._prepare_data_matrices(df, predictions_df)
        
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_assets,), dtype=np.float32
        )
        
        self.per_stock_dim = len(self.market_feature_cols) + 4 + 4
        self.global_dim = 9
        total_obs_dim = (self.num_assets * self.per_stock_dim) + self.global_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32
        )
        
        self.reset()

    def _prepare_data_matrices(self, df: pd.DataFrame, predictions_df: pd.DataFrame):
        merged = pd.merge(df, predictions_df[['ticker', 'date', 'pred_return']], on=['ticker', 'date'], how='left')
        merged['pred_return'] = merged['pred_return'].fillna(0.0)
        
        if self.scaler is not None and self.feature_cols is not None:
            close_idx = self.feature_cols.index('close')
            dummy = np.zeros((len(merged), len(self.feature_cols)))
            dummy[:, close_idx] = merged['close'].values
            merged['real_close'] = self.scaler.inverse_transform(dummy)[:, close_idx]
        else:
            merged['real_close'] = merged['close']
        
        self.price_matrix = np.zeros((self.total_steps, self.num_assets), dtype=np.float32)
        self.pred_return_matrix = np.zeros((self.total_steps, self.num_assets), dtype=np.float32)
        self.volatility_matrix = np.zeros((self.total_steps, self.num_assets), dtype=np.float32)
        self.features_matrix = np.zeros((self.total_steps, self.num_assets, len(self.market_feature_cols)), dtype=np.float32)

        for t_idx, date in enumerate(self.dates):
            date_data = merged[merged['date'] == date].set_index('ticker')
            for a_idx, ticker in enumerate(self.tickers):
                if ticker in date_data.index:
                    row = date_data.loc[ticker]
                    self.price_matrix[t_idx, a_idx] = max(row['real_close'], 1e-4)
                    self.pred_return_matrix[t_idx, a_idx] = row['pred_return']
                    self.volatility_matrix[t_idx, a_idx] = max(row.get('rolling_std_20', 0.01), 1e-4) 
                    for f_idx, col in enumerate(self.market_feature_cols):
                        self.features_matrix[t_idx, a_idx, f_idx] = row.get(col, 0.0)
                else:
                    if t_idx > 0:
                        self.price_matrix[t_idx, a_idx] = self.price_matrix[t_idx-1, a_idx]
                        self.pred_return_matrix[t_idx, a_idx] = 0.0
                        self.volatility_matrix[t_idx, a_idx] = self.volatility_matrix[t_idx-1, a_idx]
                        self.features_matrix[t_idx, a_idx] = self.features_matrix[t_idx-1, a_idx]
                    else:
                        self.price_matrix[t_idx, a_idx] = 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 60 
        self.cash = self.initial_cash
        self.stock_positions = np.zeros(self.num_assets, dtype=np.float32)
        self.entry_prices = np.zeros(self.num_assets, dtype=np.float32)
        self.holding_durations = np.zeros(self.num_assets, dtype=np.float32)
        self.prev_weights = np.zeros(self.num_assets, dtype=np.float32)
        
        self.portfolio_values = [self.initial_cash] * 60
        self.portfolio_returns = [0.0] * 60
        self.max_portfolio_value = self.initial_cash
        self.current_turnover_pct = 0.0
        
        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray):
        current_prices = self.price_matrix[self.current_step]
        current_portfolio_value = max(self.cash + np.sum(self.stock_positions * current_prices), 1e-4)
        
        raw_weights = np.clip(action, self.min_stock_weight, self.max_stock_weight)
        weight_sum = np.sum(raw_weights)
        
        if weight_sum > self.max_total_exposure:
            target_weights = (raw_weights / weight_sum) * self.max_total_exposure
        else:
            target_weights = raw_weights
            
        target_asset_values = current_portfolio_value * target_weights
        target_positions = target_asset_values / (current_prices + 1e-8)
        
        position_changes = target_positions - self.stock_positions
        trade_volumes = np.abs(position_changes) * current_prices
        
        total_turnover = np.sum(trade_volumes)
        self.current_turnover_pct = total_turnover / current_portfolio_value
        
        total_cost_rate = self.transaction_cost_pct + self.slippage_pct
        transaction_costs = total_turnover * total_cost_rate
        
        self.prev_weights = (self.stock_positions * current_prices) / current_portfolio_value
        self.stock_positions = target_positions
        self.cash = current_portfolio_value - np.sum(target_asset_values) - transaction_costs
        
        active_mask = target_positions > 1e-5
        self.entry_prices[active_mask] = np.where(self.entry_prices[active_mask] == 0, current_prices[active_mask], self.entry_prices[active_mask])
        self.entry_prices[~active_mask] = 0.0
        self.holding_durations[active_mask] += 1
        self.holding_durations[~active_mask] = 0
        
        self.current_step += 1
        next_prices = self.price_matrix[self.current_step]
        new_portfolio_value = self.cash + np.sum(self.stock_positions * next_prices)
        
        # ---> THE LOGIC FIX: Check for Bankruptcy immediately! <---
        if new_portfolio_value <= 0.20 * self.initial_cash: # If we lose 80% of our money
            self.portfolio_values.append(new_portfolio_value)
            self.portfolio_returns.append(-1.0)
            
            # Hit them with the maximum penalty and end the episode
            clip_val = self.reward_cfg.get('reward_clip', 10.0)
            return self._get_observation(), float(-clip_val), True, False, self._get_info()
            
        step_return = (new_portfolio_value - current_portfolio_value) / current_portfolio_value
        self.portfolio_values.append(new_portfolio_value)
        self.portfolio_returns.append(step_return)
        
        if new_portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = new_portfolio_value
        drawdown = (new_portfolio_value - self.max_portfolio_value) / self.max_portfolio_value
        
        log_return = np.log(new_portfolio_value / current_portfolio_value)
        rolling_vol = np.std(self.portfolio_returns[-20:])
        
        R_ret = log_return * self.reward_cfg.get('log_return_reward_lambda', 1.0)
        R_vol = rolling_vol * self.reward_cfg.get('volatility_penalty_lambda', 0.10)
        
        dd_thresh = self.reward_cfg.get('drawdown_threshold', 0.05)
        sev_dd_thresh = self.reward_cfg.get('severe_drawdown_threshold', 0.10)
        R_dd = max(0.0, abs(drawdown) - dd_thresh) * self.reward_cfg.get('drawdown_penalty_lambda', 0.20)
        R_sev_dd = max(0.0, abs(drawdown) - sev_dd_thresh) * self.reward_cfg.get('severe_drawdown_penalty_lambda', 0.40)
        
        R_cost = (transaction_costs / current_portfolio_value) * self.reward_cfg.get('transaction_cost_lambda', 1.0)
        R_turn = self.current_turnover_pct * self.reward_cfg.get('turnover_penalty_lambda', 0.05)
        
        max_alloc = np.max(target_weights)
        R_conc = max(0.0, max_alloc - 0.25) * self.reward_cfg.get('concentration_penalty_lambda', 0.05)
        
        reward = R_ret - R_vol - R_dd - R_sev_dd - R_cost - R_turn - R_conc
        
        if self.reward_cfg.get('normalize_reward', True):
            clip_val = self.reward_cfg.get('reward_clip', 10.0)
            reward = np.clip(reward * 100, -clip_val, clip_val)

        truncated = bool(self.current_step >= self.total_steps - 2)
        
        return self._get_observation(), float(reward), False, truncated, self._get_info()

    def _get_observation(self) -> np.ndarray:
        t = self.current_step
        prices = self.price_matrix[t]
        pred_returns = self.pred_return_matrix[t]
        vols = self.volatility_matrix[t]
        
        current_portfolio_val = max(self.cash + np.sum(self.stock_positions * prices), 1e-4)
        
        per_stock_obs = []
        for i in range(self.num_assets):
            m_feat = self.features_matrix[t, i]
            pred_r = pred_returns[i]
            actual_r = (prices[i] - self.price_matrix[t-1, i]) / (self.price_matrix[t-1, i] + 1e-8)
            residual = pred_r - actual_r
            risk_adj_pred = pred_r / vols[i]
            direction = 1.0 if pred_r > 0 else -1.0
            
            weight = (self.stock_positions[i] * prices[i]) / current_portfolio_val
            prev_wt = self.prev_weights[i]
            pnl = (prices[i] - self.entry_prices[i]) / (self.entry_prices[i] + 1e-8) if self.entry_prices[i] > 0 else 0.0
            duration = self.holding_durations[i] / 252.0
            
            stock_vec = np.concatenate([
                m_feat, 
                [pred_r, residual, risk_adj_pred, direction],
                [weight, prev_wt, pnl, duration]
            ])
            per_stock_obs.append(stock_vec)
            
        cash_ratio = self.cash / current_portfolio_val
        ret_1d = self.portfolio_returns[-1]
        ret_5d = (current_portfolio_val / self.portfolio_values[-5]) - 1 if len(self.portfolio_values) > 5 else 0.0
        ret_20d = (current_portfolio_val / self.portfolio_values[-20]) - 1 if len(self.portfolio_values) > 20 else 0.0
        
        vol_20 = np.std(self.portfolio_returns[-20:])
        sharpe_20 = (np.mean(self.portfolio_returns[-20:]) / (vol_20 + 1e-8)) * np.sqrt(252)
        drawdown = (current_portfolio_val - self.max_portfolio_value) / self.max_portfolio_value
        exposure = np.sum([(self.stock_positions[i] * prices[i]) / current_portfolio_val for i in range(self.num_assets)])
        
        global_obs = np.array([
            cash_ratio, ret_1d, ret_5d, ret_20d, vol_20, sharpe_20, drawdown, exposure, self.current_turnover_pct
        ], dtype=np.float32)
        
        full_obs = np.concatenate([np.concatenate(per_stock_obs), global_obs])
        full_obs = np.nan_to_num(full_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        full_obs = np.clip(full_obs, -20.0, 20.0)
        
        return full_obs.astype(np.float32)

    def _get_info(self) -> dict:
        val = self.portfolio_values[-1]
        dd = (val - self.max_portfolio_value) / self.max_portfolio_value
        return {
            'step': self.current_step,
            'portfolio_value': val,
            'cash': self.cash,
            'drawdown': dd,
            'turnover_pct': self.current_turnover_pct,
            'num_positions': np.sum(self.stock_positions > 0)
        }