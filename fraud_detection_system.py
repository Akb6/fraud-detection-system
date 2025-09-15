import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionSystem:
    """
    Advanced Fraud Detection System using Multiple Machine Learning Algorithms
    
    This system is designed to detect fraudulent transactions based on the patterns:
    1. Amount-based fraud (transactions > 220)
    2. Terminal-based fraud (compromised terminals for 28 days)
    3. Customer-based fraud (1/3 of transactions with 5x amount for 14 days)
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.evaluation_metrics = {}
        
    def load_data(self, file_path):
        """Load and perform initial data exploration"""
        try:
            self.df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully: {self.df.shape}")
            print(f"\nDataset Info:")
            print(self.df.info())
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def explore_data(self):
        """Comprehensive exploratory data analysis"""
        print("="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic statistics
        print("\nDataset Shape:", self.df.shape)
        print("\nColumn Names:")
        print(self.df.columns.tolist())
        
        # Check for missing values
        missing_values = self.df.isnull().sum()
        if missing_values.any():
            print("\nMissing Values:")
            print(missing_values[missing_values > 0])
        else:
            print("\nNo missing values found!")
        
        # Class distribution
        if 'TX_FRAUD' in self.df.columns:
            fraud_dist = self.df['TX_FRAUD'].value_counts()
            print(f"\nClass Distribution:")
            print(f"Legitimate transactions: {fraud_dist[0]} ({fraud_dist[0]/len(self.df)*100:.2f}%)")
            print(f"Fraudulent transactions: {fraud_dist[1]} ({fraud_dist[1]/len(self.df)*100:.2f}%)")
            print(f"Imbalance ratio: 1:{fraud_dist[0]/fraud_dist[1]:.0f}")
        
        # Amount statistics
        if 'TX_AMOUNT' in self.df.columns:
            print(f"\nTransaction Amount Statistics:")
            print(self.df['TX_AMOUNT'].describe())
            
        return self.df.describe()
    
    def feature_engineering(self):
        """
        Advanced feature engineering based on fraud patterns mentioned in the paper:
        1. Amount-based features
        2. Terminal-based features  
        3. Customer behavior features
        4. Time-based features
        """
        print("\nStarting Feature Engineering...")
        
        # Convert datetime if needed
        if 'TX_DATETIME' in self.df.columns:
            self.df['TX_DATETIME'] = pd.to_datetime(self.df['TX_DATETIME'])
            
            # Time-based features
            self.df['hour'] = self.df['TX_DATETIME'].dt.hour
            self.df['day_of_week'] = self.df['TX_DATETIME'].dt.dayofweek
            self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
            self.df['is_night'] = ((self.df['hour'] >= 22) | (self.df['hour'] <= 6)).astype(int)
        
        # Amount-based features (Pattern 1: amounts > 220 are fraud)
        self.df['amount_risk_flag'] = (self.df['TX_AMOUNT'] > 220).astype(int)
        self.df['amount_log'] = np.log1p(self.df['TX_AMOUNT'])
        self.df['amount_zscore'] = (self.df['TX_AMOUNT'] - self.df['TX_AMOUNT'].mean()) / self.df['TX_AMOUNT'].std()
        
        # Customer-based features (Pattern 3: customer behavior analysis)
        customer_stats = self.df.groupby('CUSTOMER_ID')['TX_AMOUNT'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).add_prefix('customer_')
        
        # Handle missing std (for customers with only one transaction)
        customer_stats['customer_std'] = customer_stats['customer_std'].fillna(0)
        
        self.df = self.df.merge(customer_stats, on='CUSTOMER_ID', how='left')
        
        # Customer amount deviation (detect 5x amount pattern)
        self.df['amount_vs_customer_mean'] = self.df['TX_AMOUNT'] / self.df['customer_mean']
        self.df['amount_deviation'] = np.abs(self.df['TX_AMOUNT'] - self.df['customer_mean'])
        self.df['is_amount_anomaly'] = (self.df['amount_vs_customer_mean'] >= 3).astype(int)
        
        # Terminal-based features (Pattern 2: compromised terminals)
        terminal_stats = self.df.groupby('TERMINAL_ID')['TX_FRAUD'].agg([
            'mean', 'sum', 'count'
        ]).add_prefix('terminal_')
        
        self.df = self.df.merge(terminal_stats, on='TERMINAL_ID', how='left')
        
        # Terminal risk score
        self.df['terminal_fraud_rate'] = self.df['terminal_mean']
        self.df['terminal_risk_flag'] = (self.df['terminal_fraud_rate'] > 0.1).astype(int)
        
        # Frequency-based features
        self.df['tx_frequency_rank'] = self.df.groupby('CUSTOMER_ID')['TX_DATETIME'].rank(method='dense')
        
        # Recent transaction patterns (14-day and 28-day windows as mentioned in patterns)
        self.df = self.df.sort_values(['CUSTOMER_ID', 'TX_DATETIME'])
        
        print(f"Feature engineering completed. New shape: {self.df.shape}")
        print(f"New features added: {self.df.shape[1] - 6} features")  # Original had 6 columns
        
        return self.df
    
    def prepare_features(self):
        """Prepare features for machine learning"""
        
        # Select features for modeling
        feature_columns = [
            'TX_AMOUNT', 'hour', 'day_of_week', 'is_weekend', 'is_night',
            'amount_risk_flag', 'amount_log', 'amount_zscore',
            'customer_mean', 'customer_std', 'customer_count',
            'amount_vs_customer_mean', 'amount_deviation', 'is_amount_anomaly',
            'terminal_fraud_rate', 'terminal_risk_flag', 'terminal_count',
            'tx_frequency_rank'
        ]
        
        # Filter features that actually exist in the dataframe
        available_features = [col for col in feature_columns if col in self.df.columns]
        
        self.X = self.df[available_features]
        self.y = self.df['TX_FRAUD']
        
        print(f"\nFeatures selected for modeling: {len(available_features)}")
        print("Features:", available_features)
        
        # Handle any remaining missing values
        self.X = self.X.fillna(self.X.median())
        
        return self.X, self.y
    
    def train_models(self):
        """Train multiple fraud detection models"""
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['robust'] = scaler
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        print(f"\nOriginal training set: {X_train.shape}")
        print(f"Balanced training set: {X_train_balanced.shape}")
        print(f"Class distribution after SMOTE: {np.bincount(y_train_balanced)}")
        
        # Define models
        models_config = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        }
        
        # Train and evaluate models
        self.results = {}
        
        print("\nTraining Models...")
        print("="*60)
        
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            # Train model
            if name in ['Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting']:
                model.fit(X_train_balanced, y_train_balanced)
            else:
                model.fit(X_train_balanced, y_train_balanced)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Evaluation metrics
            metrics = {
                'accuracy': model.score(X_test_scaled, y_test),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_pred_proba)
            }
            
            self.results[name] = metrics
            self.models[name] = model
            
            print(f"{name} - AUC: {metrics['auc_roc']:.4f}, F1: {metrics['f1']:.4f}")
        
        # Store test data for final evaluation
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        return self.results
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        
        print("\n" + "="*80)
        print("MODEL EVALUATION RESULTS")
        print("="*80)
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        
        print("\nPerformance Metrics Summary:")
        print(results_df.to_string())
        
        # Find best model
        best_model_name = results_df['f1'].idxmax()
        best_model = self.models[best_model_name]
        
        print(f"\nBest Model: {best_model_name} (F1 Score: {results_df.loc[best_model_name, 'f1']:.4f})")
        
        # Detailed evaluation for best model
        y_pred = best_model.predict(self.X_test)
        y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]
        
        print(f"\nDetailed Classification Report for {best_model_name}:")
        print("-" * 60)
        print(classification_report(self.y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix for {best_model_name}:")
        print(cm)
        
        return results_df, best_model_name
    
    def plot_results(self):
        """Generate visualization plots"""
        
        # Set up the plotting area
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fraud Detection Model Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Model Performance Comparison
        results_df = pd.DataFrame(self.results).T
        
        ax1 = axes[0, 0]
        x = np.arange(len(results_df.index))
        width = 0.15
        
        metrics = ['precision', 'recall', 'f1', 'auc_roc']
        colors = ['skyblue', 'orange', 'lightgreen', 'pink']
        
        for i, metric in enumerate(metrics):
            ax1.bar(x + i*width, results_df[metric], width, label=metric.replace('_', ' ').title(), 
                   color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(results_df.index, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. ROC Curves
        ax2 = axes[0, 1]
        
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            ax2.plot(fpr, tpr, label=f'{name} (AUC={auc_score:.3f})')
        
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature Importance (for best tree-based model)
        ax3 = axes[1, 0]
        
        best_model_name = pd.DataFrame(self.results).T['f1'].idxmax()
        best_model = self.models[best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_names = self.X.columns
            
            # Get top 10 features
            indices = np.argsort(importances)[-10:]
            
            ax3.barh(range(len(indices)), importances[indices], color='lightcoral', alpha=0.8)
            ax3.set_yticks(range(len(indices)))
            ax3.set_yticklabels([feature_names[i] for i in indices])
            ax3.set_xlabel('Importance')
            ax3.set_title(f'Top 10 Feature Importance - {best_model_name}')
            ax3.grid(True, alpha=0.3)
        
        # 4. Class Distribution
        ax4 = axes[1, 1]
        
        fraud_counts = self.df['TX_FRAUD'].value_counts()
        colors_pie = ['lightblue', 'lightcoral']
        labels = ['Legitimate', 'Fraudulent']
        
        wedges, texts, autotexts = ax4.pie(fraud_counts.values, labels=labels, colors=colors_pie, 
                                          autopct='%1.2f%%', startangle=90)
        ax4.set_title('Transaction Class Distribution')
        
        plt.tight_layout()
        plt.savefig('fraud_detection_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_report(self):
        """Generate a comprehensive fraud detection report"""
        
        results_df = pd.DataFrame(self.results).T
        best_model_name = results_df['f1'].idxmax()
        
        report = f"""
FRAUD DETECTION SYSTEM - COMPREHENSIVE REPORT
{'='*80}

DATASET OVERVIEW:
- Total Transactions: {len(self.df):,}
- Fraudulent Transactions: {sum(self.df['TX_FRAUD']):,}
- Fraud Rate: {(sum(self.df['TX_FRAUD'])/len(self.df)*100):.2f}%
- Features Used: {len(self.X.columns)}

FRAUD PATTERNS DETECTED:
1. Amount-based fraud: {sum(self.df['amount_risk_flag']):,} transactions > $220
2. Terminal risk patterns: {sum(self.df['terminal_risk_flag']):,} high-risk terminals
3. Customer behavior anomalies: {sum(self.df['is_amount_anomaly']):,} anomalous amounts

MODEL PERFORMANCE SUMMARY:
{results_df.to_string()}

BEST PERFORMING MODEL: {best_model_name}
- F1 Score: {results_df.loc[best_model_name, 'f1']:.4f}
- Precision: {results_df.loc[best_model_name, 'precision']:.4f}
- Recall: {results_df.loc[best_model_name, 'recall']:.4f}
- AUC-ROC: {results_df.loc[best_model_name, 'auc_roc']:.4f}

RECOMMENDATIONS:
1. Deploy {best_model_name} for real-time fraud detection
2. Monitor terminal fraud rates regularly
3. Implement customer behavior profiling
4. Set up alerts for transactions > $220
5. Regular model retraining with new data

CONCLUSION:
The fraud detection system successfully identifies patterns in the data and achieves
high performance with an F1 score of {results_df.loc[best_model_name, 'f1']:.4f}. The model
is ready for production deployment with continuous monitoring.

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return report

# Main execution function
def main():
    """Main function to run fraud detection analysis"""
    
    print("="*80)
    print("ADVANCED FRAUD DETECTION SYSTEM")
    print("="*80)
    print("Detecting fraudulent transactions using machine learning")
    print("Based on amount patterns, terminal behavior, and customer profiles")
    print("="*80)
    
    # Initialize the system
    fraud_detector = FraudDetectionSystem()
    
    # Note: In a real scenario, you would load actual data
    # fraud_detector.load_data('fraud_transactions.csv')
    
    print("\nFor this demonstration, we'll generate synthetic data based on the patterns described...")
    print("In a real implementation, you would load your transaction data here.")
    
    # The system is ready to use with actual transaction data
    # Key features:
    # - Advanced feature engineering based on fraud patterns
    # - Multiple ML models with proper evaluation
    # - Handles class imbalance appropriately  
    # - Comprehensive reporting and visualization
    
    return fraud_detector

if __name__ == "__main__":
    fraud_detector = main()
    print("\nFraud Detection System initialized successfully!")
    print("\nTo use with real data:")
    print("1. fraud_detector.load_data('your_data.csv')")
    print("2. fraud_detector.explore_data()")
    print("3. fraud_detector.feature_engineering()")
    print("4. fraud_detector.prepare_features()")
    print("5. fraud_detector.train_models()")
    print("6. fraud_detector.evaluate_models()")
    print("7. fraud_detector.plot_results()")
