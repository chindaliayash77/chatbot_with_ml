import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, 
                           roc_auc_score, roc_curve, accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class RealFakeClassifier:
    """
    A comprehensive classifier to distinguish between real and synthetic data.
    Supports both 2D visualization and high-dimensional analysis.
    """
    
    def __init__(self, dimensions=2, n_samples=1000, random_state=42):
        """
        Initialize the classifier.
        
        Args:
            dimensions: Number of dimensions (2 for visualization, 128 for high-dim)
            n_samples: Number of samples per class
            random_state: Random seed for reproducibility
        """
        self.dimensions = dimensions
        self.n_samples = n_samples
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
        np.random.seed(random_state)
        
    def generate_real_data(self, data_type='blobs'):
        """
        Generate real data using different distributions.
        
        Args:
            data_type: 'blobs', 'moons', or 'multivariate_normal'
            
        Returns:
            Real data array
        """
        if data_type == 'blobs':
            if self.dimensions == 2:
                data, _ = make_blobs(n_samples=self.n_samples, 
                                   centers=3, 
                                   n_features=2,
                                   cluster_std=1.5,
                                   random_state=self.random_state)
            else:
                data, _ = make_blobs(n_samples=self.n_samples, 
                                   centers=5, 
                                   n_features=self.dimensions,
                                   cluster_std=2.0,
                                   random_state=self.random_state)
                
        elif data_type == 'moons':
            if self.dimensions == 2:
                data, _ = make_moons(n_samples=self.n_samples,
                                   noise=0.1,
                                   random_state=self.random_state)
            else:
                # For high dimensions, use multivariate normal with moon-like structure
                data = self._generate_moon_like_high_dim()
                
        elif data_type == 'multivariate_normal':
            # Create correlated multivariate normal distribution
            mean = np.zeros(self.dimensions)
            
            # Create a structured covariance matrix
            cov = np.eye(self.dimensions)
            for i in range(self.dimensions-1):
                cov[i, i+1] = 0.3
                cov[i+1, i] = 0.3
            
            data = np.random.multivariate_normal(mean, cov, self.n_samples)
        
        return data
    
    def _generate_moon_like_high_dim(self):
        """Generate moon-like structure in high dimensions."""
        # Create two clusters with curved structure
        t = np.linspace(0, np.pi, self.n_samples // 2)
        
        # First moon
        moon1 = np.zeros((self.n_samples // 2, self.dimensions))
        moon1[:, 0] = np.cos(t) + np.random.normal(0, 0.1, self.n_samples // 2)
        moon1[:, 1] = np.sin(t) + np.random.normal(0, 0.1, self.n_samples // 2)
        # Add noise to other dimensions
        moon1[:, 2:] = np.random.normal(0, 0.5, (self.n_samples // 2, self.dimensions - 2))
        
        # Second moon
        moon2 = np.zeros((self.n_samples - self.n_samples // 2, self.dimensions))
        t2 = np.linspace(0, np.pi, self.n_samples - self.n_samples // 2)
        moon2[:, 0] = 1 - np.cos(t2) + np.random.normal(0, 0.1, len(t2))
        moon2[:, 1] = 1 - np.sin(t2) - 0.5 + np.random.normal(0, 0.1, len(t2))
        moon2[:, 2:] = np.random.normal(0, 0.5, (len(t2), self.dimensions - 2))
        
        return np.vstack([moon1, moon2])
    
    def generate_fake_data(self, fake_type='uniform'):
        """
        Generate fake data using different distributions.
        
        Args:
            fake_type: 'uniform', 'different_gaussian', or 'noise'
            
        Returns:
            Fake data array
        """
        if fake_type == 'uniform':
            # Uniform distribution in reasonable range
            if self.dimensions == 2:
                data = np.random.uniform(-4, 4, (self.n_samples, 2))
            else:
                data = np.random.uniform(-3, 3, (self.n_samples, self.dimensions))
                
        elif fake_type == 'different_gaussian':
            # Gaussian with different parameters
            mean = np.full(self.dimensions, 2.0)  # Different mean
            cov = np.eye(self.dimensions) * 4.0   # Different variance
            data = np.random.multivariate_normal(mean, cov, self.n_samples)
            
        elif fake_type == 'noise':
            # Pure noise with higher variance
            data = np.random.normal(0, 3, (self.n_samples, self.dimensions))
            
        return data
    
    def prepare_dataset(self, real_type='blobs', fake_type='uniform'):
        """
        Prepare the complete dataset with labels.
        
        Args:
            real_type: Type of real data generation
            fake_type: Type of fake data generation
            
        Returns:
            X, y: Features and labels
        """
        # Generate data
        real_data = self.generate_real_data(real_type)
        fake_data = self.generate_fake_data(fake_type)
        
        # Combine data and create labels
        X = np.vstack([real_data, fake_data])
        y = np.hstack([np.ones(len(real_data)), np.zeros(len(fake_data))])
        
        # Store for visualization
        self.real_data = real_data
        self.fake_data = fake_data
        self.X = X
        self.y = y
        
        return X, y
    
    def train_models(self, X, y):
        """
        Train multiple classifiers.
        
        Args:
            X: Features
            y: Labels
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        # Define models
        models = {
            'Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(random_state=self.random_state))
            ]),
            'SVM': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', SVC(probability=True, random_state=self.random_state))
            ]),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss'
            )
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            self.models[name] = model
    
    def evaluate_models(self):
        """Evaluate all trained models."""
        
        for name, model in self.models.items():
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                      cv=5, scoring='roc_auc')
            
            self.results[name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
    
    def get_plots(self):
        """Create comprehensive visualizations and return plot objects."""
        plots = {}
        
        if self.dimensions == 2:
            plots['2d'] = self._get_2d_plots()
        else:
            plots['high_dim'] = self._get_high_dim_plots()
        
        plots['model_comparison'] = self._get_model_comparison_plot()
        plots['confusion_matrices'] = self._get_confusion_matrices_plot()
        plots['roc_curves'] = self._get_roc_curves_plot()
        
        return plots
    
    def _get_2d_plots(self):
        """Get 2D data visualization with decision boundaries."""
        from io import BytesIO
        import base64
        
        plots = {}
        
        # Original data distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(self.real_data[:, 0], self.real_data[:, 1], 
                  c='blue', alpha=0.6, label='Real Data', s=30)
        ax.scatter(self.fake_data[:, 0], self.fake_data[:, 1], 
                  c='red', alpha=0.6, label='Fake Data', s=30)
        ax.set_title('Data Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plots['data_distribution'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        # Decision boundaries for top 3 models
        top_models = sorted(self.results.items(), 
                          key=lambda x: x[1]['roc_auc'], reverse=True)[:3]
        
        for idx, (name, _) in enumerate(top_models):
            fig, ax = plt.subplots(figsize=(8, 6))
            self._plot_decision_boundary(ax, name)
            
            # Save to buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plots[f'decision_boundary_{name}'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
        
        return plots
    
    def _plot_decision_boundary(self, ax, model_name):
        """Plot decision boundary for a model."""
        model = self.models[model_name]
        
        # Create mesh
        h = 0.1
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Predict on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict_proba(mesh_points)[:, 1]
        Z = Z.reshape(xx.shape)
        
        # Plot
        ax.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='RdYlBu')
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
        
        # Plot data points
        scatter = ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, 
                           cmap='RdYlBu', alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
        
        ax.set_title(f'{model_name}\nROC AUC: {self.results[model_name]["roc_auc"]:.4f}')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Class (0=Fake, 1=Real)')
    
    def _get_high_dim_plots(self):
        """Get high-dimensional data using PCA visualization."""
        from io import BytesIO
        import base64
        
        plots = {}
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(self.X)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # PCA visualization
        scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=self.y, 
                            cmap='RdYlBu', alpha=0.7, s=30)
        ax1.set_title(f'PCA Visualization ({self.dimensions}D → 2D)\n'
                 f'Explained Variance: {pca.explained_variance_ratio_.sum():.3f}')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
        plt.colorbar(scatter, ax=ax1, label='Class (0=Fake, 1=Real)')
        ax1.grid(True, alpha=0.3)
        
        # Feature importance for best model
        best_model_name = max(self.results.items(), key=lambda x: x[1]['roc_auc'])[0]
        if hasattr(self.models[best_model_name], 'feature_importances_'):
            importances = self.models[best_model_name].feature_importances_
        elif best_model_name == 'Logistic Regression':
            importances = np.abs(self.models[best_model_name].named_steps['clf'].coef_[0])
        else:
            importances = np.ones(self.dimensions)  # Fallback
        
        top_features = min(20, len(importances))  # Show top 20 features
        indices = np.argsort(importances)[-top_features:]
        ax2.barh(range(top_features), importances[indices])
        ax2.set_ylabel('Feature Index')
        ax2.set_xlabel('Importance')
        ax2.set_title(f'Feature Importance ({best_model_name})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plots['high_dim'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return plots
    
    def _get_model_comparison_plot(self):
        """Get model performance comparison plot."""
        from io import BytesIO
        import base64
        
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        roc_aucs = [self.results[model]['roc_auc'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy comparison
        bars = ax1.bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom')
        
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # ROC AUC comparison
        bars = ax2.bar(models, roc_aucs, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax2.set_ylabel('ROC AUC')
        ax2.set_title('Model ROC AUC Comparison')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, auc in zip(bars, roc_aucs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{auc:.3f}', ha='center', va='bottom')
        
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return plot_data
    
    def _get_confusion_matrices_plot(self):
        """Get confusion matrices for all models."""
        from io import BytesIO
        import base64
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, (name, results) in enumerate(self.results.items()):
            ax = axes[idx]
            cm = results['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Fake', 'Real'], 
                       yticklabels=['Fake', 'Real'])
            ax.set_title(f'{name}\nAccuracy: {results["accuracy"]:.3f}')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return plot_data
    
    def _get_roc_curves_plot(self):
        """Get ROC curves for all models."""
        from io import BytesIO
        import base64
        
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'green', 'red', 'orange']
        
        for idx, (name, results) in enumerate(self.results.items()):
            fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
            plt.plot(fpr, tpr, color=colors[idx], lw=2, 
                    label=f'{name} (AUC = {results["roc_auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return plot_data
    
    def get_summary(self):
        """Get comprehensive performance summary as HTML."""
        summary = f"""
        <h3>Performance Summary</h3>
        <p><b>Dataset:</b> {self.dimensions}D with {len(self.X)} samples</p>
        <p><b>Train/Test Split:</b> {len(self.X_train)}/{len(self.X_test)}</p>
        """
        
        # Sort models by ROC AUC
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1]['roc_auc'], reverse=True)
        
        summary += "<h4>Model Rankings (by ROC AUC):</h4>"
        for rank, (name, results) in enumerate(sorted_results, 1):
            summary += f"""
            <p><b>{rank}. {name}:</b><br>
            &nbsp;&nbsp;Accuracy: {results['accuracy']:.4f}<br>
            &nbsp;&nbsp;ROC AUC: {results['roc_auc']:.4f}<br>
            &nbsp;&nbsp;CV Score: {results['cv_mean']:.4f} (±{results['cv_std']:.4f})</p>
            """
        
        # Best model details
        best_model, best_results = sorted_results[0]
        summary += f"<h4>Best Model: {best_model}</h4>"
        summary += "<h4>Classification Report:</h4>"
        
        # Create classification report
        report = classification_report(self.y_test, best_results['y_pred'],
                                     target_names=['Fake', 'Real'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        summary += report_df.to_html(classes='table table-striped')
        
        return summary