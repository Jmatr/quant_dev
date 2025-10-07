import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

from core.strategy import Strategy
from core.feature_engineering import feature_engineer


class MLStrategy(Strategy):
    """Machine Learning Based Trading Strategy with Walk-Forward Analysis"""

    def __init__(self, model_type: str = 'random_forest', lookforward_days: int = 5,
                 retrain_freq: int = 21, min_train_size: float = 0.2):
        super().__init__(f"ML Strategy ({model_type})")
        self.model_type = model_type
        self.lookforward_days = lookforward_days
        self.retrain_freq = retrain_freq  # 重训练频率（交易日）
        self.min_train_size_ratio = min_train_size  # 最小训练集比例
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.training_history = []  # 记录训练历史

    def prepare_data(self):
        """Prepare data for machine learning"""
        if self.data is None:
            return self

        # Create features and target
        self.features, self.target = feature_engineer.prepare_ml_dataset(
            self.data, self.lookforward_days
        )

        print(f"ML Dataset: {len(self.features)} samples, {len(self.features.columns)} features")
        return self

    def _create_model(self):
        """创建新的模型实例"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'logistic':
            return LogisticRegression(random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def walk_forward_train(self):
        """Walk-Forward训练和分析"""
        if self.features is None or self.target is None:
            raise ValueError("No features or target available. Call prepare_data() first.")

        signals = pd.Series(0, index=self.features.index)

        # 计算最小训练集大小
        min_train_size = int(len(self.features) * self.min_train_size_ratio)
        min_train_size = max(min_train_size, 100)  # 至少100个样本

        print(f"Walk-Forward Analysis: 最小训练集 {min_train_size} 样本, 重训练频率 {self.retrain_freq} 天")

        # Walk-Forward循环
        total_predictions = 0
        total_buy_signals = 0

        for i in range(min_train_size, len(self.features), self.retrain_freq):
            # 训练数据截止到当前
            train_end = i
            X_train = self.features.values[:train_end]
            y_train = self.target.values[:train_end]

            # 预测未来retrain_freq天
            test_start = train_end
            test_end = min(train_end + self.retrain_freq, len(self.features))

            if test_start >= test_end:
                continue

            # 训练模型
            model = self._create_model()
            X_train_scaled = self.scaler.fit_transform(X_train)
            model.fit(X_train_scaled, y_train)

            # 记录训练信息
            train_date = self.features.index[train_end - 1]
            self.training_history.append({
                'train_end_date': train_date,
                'train_size': len(X_train),
                'test_start_date': self.features.index[test_start],
                'test_end_date': self.features.index[test_end - 1] if test_end < len(self.features) else
                self.features.index[-1]
            })

            # 预测测试期
            X_test = self.features.values[test_start:test_end]
            X_test_scaled = self.scaler.transform(X_test)
            test_predictions = model.predict(X_test_scaled)
            test_proba = model.predict_proba(X_test_scaled)[:, 1]

            # 调试信息
            period_predictions = len(test_predictions)
            period_buy_signals = 0

            # 在测试期生成信号（调整阈值增加交易机会）
            for j in range(len(test_predictions)):
                idx = test_start + j
                if idx < len(signals):
                    # 降低置信度阈值，增加交易机会
                    if test_predictions[j] == 1 and test_proba[j] > 0.55:  # 从0.6降低到0.55
                        signals.iloc[idx] = 1
                        period_buy_signals += 1
                    elif test_predictions[j] == 0 or test_proba[j] < 0.45:  # 从0.4调整到0.45
                        signals.iloc[idx] = 0

            total_predictions += period_predictions
            total_buy_signals += period_buy_signals

            print(f"训练周期 {len(self.training_history)}: {period_buy_signals}/{period_predictions} 买入信号")

        print(
            f"总体信号统计: {total_buy_signals}/{total_predictions} 买入信号 ({total_buy_signals / total_predictions * 100:.1f}%)")

        # 添加必要的shift避免前视偏差
        signals = signals.shift(self.lookforward_days).fillna(0)

        # 确保信号长度匹配
        aligned_signals = pd.Series(0, index=self.data.index)
        common_dates = aligned_signals.index.intersection(signals.index)
        aligned_signals.loc[common_dates] = signals.loc[common_dates]

        self.signals = aligned_signals

        print(f"Walk-Forward训练完成: {len(self.training_history)} 次训练")
        print(f"最终信号: {len(aligned_signals)} 个, {(aligned_signals == 1).sum()} 个买入信号")

        return self

    def generate_signals(self):
        """生成交易信号 - 使用Walk-Forward分析"""
        return self.walk_forward_train()

    def analyze_feature_importance(self):
        """分析特征重要性"""
        if hasattr(self, 'feature_importance') and self.feature_importance is not None:
            print("特征重要性分析:")
            print(self.feature_importance.head(10))

            # 检查是否有特征重要性为0
            zero_importance = self.feature_importance[
                self.feature_importance['importance'] == 0
                ]
            if len(zero_importance) > 0:
                print(f"警告: {len(zero_importance)} 个特征重要性为0")

        return self.feature_importance

    def get_training_summary(self):
        """获取训练摘要"""
        if not self.training_history:
            return "No training history available"

        df = pd.DataFrame(self.training_history)
        return df


class EnsembleMLStrategy(Strategy):
    """Ensemble of Multiple ML Models with Walk-Forward Analysis"""

    def __init__(self, lookforward_days: int = 5, retrain_freq: int = 21,
                 min_train_size: float = 0.2, voting_threshold: float = 0.55):
        super().__init__("Ensemble ML Strategy")
        self.lookforward_days = lookforward_days
        self.retrain_freq = retrain_freq
        self.min_train_size_ratio = min_train_size
        self.voting_threshold = voting_threshold  # 降低投票阈值
        self.models = {}
        self.scaler = StandardScaler()
        self.training_history = []

    def prepare_data(self):
        """Prepare data for ensemble models"""
        if self.data is None:
            return self

        self.features, self.target = feature_engineer.prepare_ml_dataset(
            self.data, self.lookforward_days
        )
        return self

    def _create_models(self):
        """创建集成模型"""
        return {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic': LogisticRegression(random_state=42)
        }

    def walk_forward_train(self):
        """Walk-Forward训练集成模型"""
        if self.features is None or self.target is None:
            raise ValueError("No features or target available.")

        signals = pd.Series(0, index=self.features.index)
        min_train_size = int(len(self.features) * self.min_train_size_ratio)
        min_train_size = max(min_train_size, 100)

        print(f"Ensemble Walk-Forward: 最小训练集 {min_train_size} 样本, 投票阈值 {self.voting_threshold}")

        total_predictions = 0
        total_buy_signals = 0

        for i in range(min_train_size, len(self.features), self.retrain_freq):
            train_end = i
            test_start = train_end
            test_end = min(train_end + self.retrain_freq, len(self.features))

            if test_start >= test_end:
                continue

            # 训练集成模型
            X_train = self.features.values[:train_end]
            y_train = self.target.values[:train_end]
            X_train_scaled = self.scaler.fit_transform(X_train)

            models = self._create_models()
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)

            # 记录训练信息
            self.training_history.append({
                'train_end_date': self.features.index[train_end - 1],
                'train_size': len(X_train)
            })

            # 集成预测
            X_test = self.features.values[test_start:test_end]
            X_test_scaled = self.scaler.transform(X_test)

            # 获取所有模型的预测概率
            proba_predictions = []
            for name, model in models.items():
                proba = model.predict_proba(X_test_scaled)[:, 1]
                proba_predictions.append(proba)

            # 平均概率
            avg_proba = np.mean(proba_predictions, axis=0)

            # 调试信息
            period_predictions = len(avg_proba)
            period_buy_signals = 0

            # 生成信号
            for j in range(len(avg_proba)):
                idx = test_start + j
                if idx < len(signals):
                    total_predictions += 1
                    if avg_proba[j] > self.voting_threshold:
                        signals.iloc[idx] = 1
                        period_buy_signals += 1
                        total_buy_signals += 1
                    else:
                        signals.iloc[idx] = 0

            print(f"Ensemble周期 {len(self.training_history)}: {period_buy_signals}/{period_predictions} 买入信号")

        print(
            f"Ensemble总体信号: {total_buy_signals}/{total_predictions} 买入信号 ({total_buy_signals / total_predictions * 100:.1f}%)")

        # 添加shift和对齐
        signals = signals.shift(self.lookforward_days).fillna(0)
        aligned_signals = pd.Series(0, index=self.data.index)
        common_dates = aligned_signals.index.intersection(signals.index)
        aligned_signals.loc[common_dates] = signals.loc[common_dates]

        self.signals = aligned_signals
        print(f"Ensemble Walk-Forward完成: {len(self.training_history)} 次训练")
        print(f"最终信号: {len(aligned_signals)} 个, {(aligned_signals == 1).sum()} 个买入信号")

        return self

    def generate_signals(self):
        """生成集成信号 - 使用Walk-Forward分析"""
        return self.walk_forward_train()