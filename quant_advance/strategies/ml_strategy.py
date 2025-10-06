import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

from core.strategy import Strategy
from core.feature_engineering import feature_engineer


class MLStrategy(Strategy):
    """Machine Learning Based Trading Strategy"""

    def __init__(self, model_type: str = 'random_forest', lookforward_days: int = 5):
        super().__init__(f"ML Strategy ({model_type})")
        self.model_type = model_type
        self.lookforward_days = lookforward_days
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None

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

    def train_model(self, test_size: float = 0.2):
        """Train machine learning model"""
        if self.features is None or self.target is None:
            raise ValueError("No features or target available. Call prepare_data() first.")

        # Split data (time series aware)
        tscv = TimeSeriesSplit(n_splits=5)

        X = self.features.values
        y = self.target.values

        # Use last portion as test set for time series
        split_point = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Model trained: {self.model_type}")
        print(f"Test Accuracy: {accuracy:.3f}")
        print(classification_report(y_test, y_pred))

        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.features.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nTop 10 Feature Importances:")
            print(self.feature_importance.head(10))

        return accuracy

    def generate_signals(self):
        """Generate trading signals using ML predictions - Fixed version"""
        if self.model is None:
            self.train_model()

        # Prepare features for prediction
        X = self.features.values
        X_scaled = self.scaler.transform(X)

        # Make predictions
        predictions = self.model.predict(X_scaled)
        prediction_proba = self.model.predict_proba(X_scaled)[:, 1]

        # Create signals based on predictions and confidence
        signals = pd.Series(0, index=self.features.index)

        # Only take long positions when model predicts positive return with high confidence
        signals[(predictions == 1) & (prediction_proba > 0.6)] = 1
        # Exit positions when confidence drops or prediction turns negative
        signals[(predictions == 0) | (prediction_proba < 0.4)] = 0

        # Ensure we don't trade on the most recent data (avoid look-ahead)
        signals = signals.shift(self.lookforward_days).fillna(0)

        # Align signals with original data index
        aligned_signals = pd.Series(0, index=self.data.index)
        aligned_signals.update(signals)

        self.signals = aligned_signals
        print(f"ML Strategy: {len(aligned_signals)} signals, {(aligned_signals == 1).sum()} buy signals")

        return self


class EnsembleMLStrategy(Strategy):
    """Ensemble of Multiple ML Models"""

    def __init__(self, lookforward_days: int = 5):
        super().__init__("Ensemble ML Strategy")
        self.lookforward_days = lookforward_days
        self.models = {}
        self.scaler = StandardScaler()
        self.voting_threshold = 0.6

    def prepare_data(self):
        """Prepare data for ensemble models"""
        if self.data is None:
            return self

        self.features, self.target = feature_engineer.prepare_ml_dataset(
            self.data, self.lookforward_days
        )
        return self

    def train_ensemble(self):
        """Train ensemble of models"""
        if self.features is None or self.target is None:
            raise ValueError("No features or target available.")

        X = self.features.values
        y = self.target.values

        split_point = int(len(X) * 0.8)
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Define models
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic': LogisticRegression(random_state=42)
        }

        # Train all models
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} accuracy: {accuracy:.3f}")

    def generate_signals(self):
        """Generate signals using ensemble voting - Fixed version"""
        if not self.models:
            self.train_ensemble()

        X = self.features.values
        X_scaled = self.scaler.transform(X)

        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict_proba(X_scaled)[:, 1]

        # Ensemble voting (average probability)
        avg_proba = np.mean(list(predictions.values()), axis=0)

        # Create signals based on ensemble consensus
        signals = pd.Series(0, index=self.features.index)
        signals[avg_proba > self.voting_threshold] = 1

        # Shift to avoid look-ahead bias
        signals = signals.shift(self.lookforward_days).fillna(0)

        # Align with original data
        aligned_signals = pd.Series(0, index=self.data.index)
        aligned_signals.update(signals)

        self.signals = aligned_signals
        print(f"Ensemble Strategy: {len(aligned_signals)} signals, {(aligned_signals == 1).sum()} buy signals")

        return self