# core/factory.py
from modules.universe.yfinance_loader import YFinanceUniverseSelector
from modules.cleaning.standard_cleaner import StandardCleaner
from modules.features.hybrid_features import HybridFeatureEngineer
from modules.clustering.hierarchical import HierarchicalClusterer
from modules.clustering.random_clusterer import RandomClusterer
from modules.priors.abm_prior import EnsembleABMPrior
from modules.priors.time_decay_prior import TimeDecayEnsemblePrior
from modules.weights.huber_weights import HuberWeightGenerator 
from modules.weights.equal_weights import EqualWeightGenerator 
from modules.predictors.lstm_cnn_mlp import LSTM_CNN_MLP

class ModuleFactory:
    @staticmethod
    def get_universe(config: dict):
        loader_type = config.get('type')
        if loader_type == "YFinance":
            return YFinanceUniverseSelector(config)
        else:
            raise ValueError(f"Unknown Universe Selector type: {loader_type}")

    @staticmethod
    def get_cleaner(config: dict):
        cleaner_type = config.get('type')
        if cleaner_type == "StandardCleaner":
            return StandardCleaner(config)
        else:
            raise ValueError(f"Unknown Cleaner type: {cleaner_type}")
    
    @staticmethod
    def get_feature_engineer(config: dict):
        feature_engineer_type = config.get('type')
        if feature_engineer_type == "HybridFeatureEngineer":
            return HybridFeatureEngineer(config)
        else:
            raise ValueError(f"Unknown Feature Engineer: {feature_engineer_type}")

    @staticmethod
    def get_clusterer(config: dict):
        clusterer_type = config.get('type')
        if clusterer_type == "HierarchicalClustering":
            return HierarchicalClusterer(config)
        elif clusterer_type == "RandomClusterer":
            return RandomClusterer(config)
        else:
            raise ValueError("Unknown Clusterer")

    @staticmethod
    def get_prior(config: dict):
        prior_type = config.get('type')
        if prior_type == "EnsembleABMPrior":
            return EnsembleABMPrior(config)
        elif prior_type == "TimeDecayEnsemblePrior":              
            return TimeDecayEnsemblePrior(config)                  
        raise ValueError(f"Unknown Prior Model: {prior_type}")

    @staticmethod
    def get_weight_generator(config: dict):
        wtype = config.get('type')
        if wtype == "HuberWeightGenerator":
            return HuberWeightGenerator(config)
        elif wtype == "EqualWeightGenerator":
            return EqualWeightGenerator(config)
        raise ValueError(f"Unknown Weight Generator: {wtype}")

    @staticmethod
    def get_predictor(config: dict):
        pred_type = config.get('type')
        if pred_type == "lstm_cnn_mlp":
            return LSTM_CNN_MLP(config)
        raise ValueError(f"Unknown Predictor Model: {pred_type}")
    
    @staticmethod
    def get_loss(config): pass
    
    @staticmethod
    def get_rl_agent(config): pass
    
    @staticmethod
    def get_evaluator(config): pass