"""
Enhanced Meta-Feature Extraction Module
Implements comprehensive meta-feature extraction including:
- Statistical features (basic + advanced)
- Information-theoretic features
- Model-based features  
- Landmarking features
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class EnhancedMetaFeatureExtractor:
    """
    Comprehensive meta-feature extraction combining multiple types of features
    as used in automated machine learning and meta-learning research.
    """
    
    def __init__(self, sample_size=3000, cv_folds=3, random_state=42):
        self.sample_size = sample_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoder = LabelEncoder()
        
    def extract_all_metafeatures(self, dataset):
        """Extract all types of meta-features from a dataset"""
        try:
            X, y = dataset['X'].copy(), dataset['y'].copy()
            
            # Handle missing values and ensure numeric encoding
            X, y = self._preprocess_data(X, y)
            
            # Sample if too large
            if len(X) > self.sample_size:
                indices = np.random.choice(len(X), self.sample_size, replace=False)
                X, y = X.iloc[indices], y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
            
            metafeatures = {}
            
            # Extract all feature types
            metafeatures.update(self._extract_statistical_features(X, y))
            metafeatures.update(self._extract_information_theoretic_features(X, y))
            metafeatures.update(self._extract_model_based_features(X, y))
            metafeatures.update(self._extract_landmarking_features(X, y))
            
            return metafeatures
            
        except Exception as e:
            print(f"Error extracting meta-features: {e}")
            return self._get_default_metafeatures()
    
    def _preprocess_data(self, X, y):
        """Preprocess data for meta-feature extraction"""
        # Handle categorical columns
        X_processed = X.copy()
        for col in X_processed.select_dtypes(include=['object', 'category']).columns:
            X_processed[col] = pd.Categorical(X_processed[col]).codes
        
        # Fill missing values
        X_processed = pd.DataFrame(
            self.imputer.fit_transform(X_processed), 
            columns=X_processed.columns
        )
        
        # Encode target if needed
        y_processed = y.copy()
        if hasattr(y, 'dtype') and y.dtype == 'object':
            y_processed = self.label_encoder.fit_transform(y)
        
        return X_processed, y_processed
    
    def _extract_statistical_features(self, X, y):
        """Extract statistical meta-features"""
        features = {}
        
        # Basic dataset characteristics
        features['n_instances'] = len(X)
        features['n_features'] = X.shape[1]
        features['n_classes'] = len(np.unique(y))
        features['n_numeric_features'] = len(X.select_dtypes(include=['number']).columns)
        features['n_categorical_features'] = X.shape[1] - features['n_numeric_features']
        
        # Ratios
        features['instances_to_features'] = features['n_instances'] / max(1, features['n_features'])
        features['features_to_instances'] = features['n_features'] / max(1, features['n_instances'])
        features['categorical_ratio'] = features['n_categorical_features'] / max(1, features['n_features'])
        
        # Class distribution features
        class_counts = np.bincount(y.astype(int))
        features['class_imbalance'] = (class_counts.max() - class_counts.min()) / max(1, class_counts.max())
        features['minority_class_ratio'] = class_counts.min() / max(1, class_counts.sum())
        features['majority_class_ratio'] = class_counts.max() / max(1, class_counts.sum())
        features['class_entropy'] = entropy(class_counts)
        
        # Statistical properties of numeric features
        numeric_cols = X.select_dtypes(include=['number'])
        if len(numeric_cols.columns) > 0:
            # Central tendency
            features['mean_mean'] = numeric_cols.mean().mean()
            features['mean_std'] = numeric_cols.std().mean()
            features['mean_skewness'] = numeric_cols.skew().mean()
            features['mean_kurtosis'] = numeric_cols.kurtosis().mean()
            
            # Variability
            features['std_mean'] = numeric_cols.mean().std()
            features['std_std'] = numeric_cols.std().std()
            
            # Correlation analysis
            corr_matrix = numeric_cols.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            features['mean_correlation'] = upper_tri.stack().mean()
            features['max_correlation'] = upper_tri.stack().max()
            features['min_correlation'] = upper_tri.stack().min()
            
        else:
            # Default values for non-numeric datasets
            for feature_name in ['mean_mean', 'mean_std', 'mean_skewness', 'mean_kurtosis',
                               'std_mean', 'std_std', 'mean_correlation', 'max_correlation', 'min_correlation']:
                features[feature_name] = 0.0
        
        return features
    
    def _extract_information_theoretic_features(self, X, y):
        """Extract information-theoretic meta-features"""
        features = {}
        
        try:
            # Target entropy
            _, counts = np.unique(y, return_counts=True)
            features['target_entropy'] = entropy(counts)
            
            # Mutual information between features and target
            if X.shape[1] > 0:
                mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
                features['mean_mutual_info'] = np.mean(mi_scores)
                features['max_mutual_info'] = np.max(mi_scores)
                features['min_mutual_info'] = np.min(mi_scores)
                features['std_mutual_info'] = np.std(mi_scores)
                
                # Normalized mutual information
                if features['target_entropy'] > 0:
                    normalized_mi = mi_scores / features['target_entropy']
                    features['mean_normalized_mi'] = np.mean(normalized_mi)
                else:
                    features['mean_normalized_mi'] = 0.0
            else:
                for feature_name in ['mean_mutual_info', 'max_mutual_info', 'min_mutual_info', 
                                   'std_mutual_info', 'mean_normalized_mi']:
                    features[feature_name] = 0.0
            
            # Equivalent number of attributes
            if features['target_entropy'] > 0 and 'mean_mutual_info' in features:
                features['equivalent_num_attributes'] = features['target_entropy'] / max(1e-8, features['mean_mutual_info'])
            else:
                features['equivalent_num_attributes'] = 0.0
                
        except Exception as e:
            print(f"Error in information-theoretic features: {e}")
            for feature_name in ['target_entropy', 'mean_mutual_info', 'max_mutual_info', 'min_mutual_info',
                               'std_mutual_info', 'mean_normalized_mi', 'equivalent_num_attributes']:
                features[feature_name] = 0.0
        
        return features
    
    def _extract_model_based_features(self, X, y):
        """Extract model-based meta-features using simple models"""
        features = {}
        
        try:
            # Ensure we have enough samples for cross-validation
            if len(X) < self.cv_folds * 2:
                cv_folds = max(2, len(X) // 2)
            else:
                cv_folds = self.cv_folds
            
            # Decision tree features
            dt = DecisionTreeClassifier(random_state=self.random_state, max_depth=10)
            dt_scores = cross_val_score(dt, X, y, cv=cv_folds, scoring='accuracy')
            features['dt_mean_accuracy'] = np.mean(dt_scores)
            features['dt_std_accuracy'] = np.std(dt_scores)
            
            # Fit tree to get depth and nodes
            dt.fit(X, y)
            features['dt_depth'] = dt.get_depth()
            features['dt_n_leaves'] = dt.get_n_leaves()
            features['dt_nodes_per_attribute'] = dt.tree_.node_count / max(1, X.shape[1])
            features['dt_nodes_per_instance'] = dt.tree_.node_count / max(1, len(X))
            
            # Linear model discriminability
            if X.shape[1] < len(X):  # Only if we have more instances than features
                lr = LogisticRegression(random_state=self.random_state, max_iter=100, solver='liblinear')
                lr_scores = cross_val_score(lr, X, y, cv=cv_folds, scoring='accuracy')
                features['lr_mean_accuracy'] = np.mean(lr_scores)
                features['lr_std_accuracy'] = np.std(lr_scores)
            else:
                features['lr_mean_accuracy'] = 0.0
                features['lr_std_accuracy'] = 0.0
                
        except Exception as e:
            print(f"Error in model-based features: {e}")
            for feature_name in ['dt_mean_accuracy', 'dt_std_accuracy', 'dt_depth', 'dt_n_leaves',
                               'dt_nodes_per_attribute', 'dt_nodes_per_instance', 
                               'lr_mean_accuracy', 'lr_std_accuracy']:
                features[feature_name] = 0.0
        
        return features
    
    def _extract_landmarking_features(self, X, y):
        """Extract landmarking meta-features using simple algorithms"""
        features = {}
        
        try:
            # Ensure we have enough samples
            if len(X) < self.cv_folds * 2:
                cv_folds = max(2, len(X) // 2)
            else:
                cv_folds = self.cv_folds
            
            # Naive Bayes landmark
            nb = GaussianNB()
            nb_scores = cross_val_score(nb, X, y, cv=cv_folds, scoring='accuracy')
            features['nb_landmark'] = np.mean(nb_scores)
            
            # 1-NN landmark
            if len(X) > cv_folds:
                knn = KNeighborsClassifier(n_neighbors=1)
                knn_scores = cross_val_score(knn, X, y, cv=cv_folds, scoring='accuracy')
                features['1nn_landmark'] = np.mean(knn_scores)
            else:
                features['1nn_landmark'] = 0.0
            
            # Random node landmark (single decision stump)
            dt_stump = DecisionTreeClassifier(max_depth=1, random_state=self.random_state)
            stump_scores = cross_val_score(dt_stump, X, y, cv=cv_folds, scoring='accuracy')
            features['random_node_landmark'] = np.mean(stump_scores)
            
            # Worst node landmark (always predict majority class)
            majority_class_prob = np.bincount(y.astype(int)).max() / len(y)
            features['worst_node_landmark'] = majority_class_prob
            
            # Linear discriminant landmark (approximated with logistic regression)
            if X.shape[1] < len(X):
                lr_simple = LogisticRegression(random_state=self.random_state, max_iter=50, solver='liblinear')
                lr_simple_scores = cross_val_score(lr_simple, X, y, cv=cv_folds, scoring='accuracy')
                features['linear_discriminant_landmark'] = np.mean(lr_simple_scores)
            else:
                features['linear_discriminant_landmark'] = 0.0
                
        except Exception as e:
            print(f"Error in landmarking features: {e}")
            for feature_name in ['nb_landmark', '1nn_landmark', 'random_node_landmark', 
                               'worst_node_landmark', 'linear_discriminant_landmark']:
                features[feature_name] = 0.0
        
        return features
    
    def _get_default_metafeatures(self):
        """Return default meta-features in case of extraction failure"""
        default_features = {}
        
        # Default statistical features
        stat_features = ['n_instances', 'n_features', 'n_classes', 'n_numeric_features', 'n_categorical_features',
                        'instances_to_features', 'features_to_instances', 'categorical_ratio',
                        'class_imbalance', 'minority_class_ratio', 'majority_class_ratio', 'class_entropy',
                        'mean_mean', 'mean_std', 'mean_skewness', 'mean_kurtosis', 'std_mean', 'std_std',
                        'mean_correlation', 'max_correlation', 'min_correlation']
        
        # Default information-theoretic features  
        info_features = ['target_entropy', 'mean_mutual_info', 'max_mutual_info', 'min_mutual_info',
                        'std_mutual_info', 'mean_normalized_mi', 'equivalent_num_attributes']
        
        # Default model-based features
        model_features = ['dt_mean_accuracy', 'dt_std_accuracy', 'dt_depth', 'dt_n_leaves',
                         'dt_nodes_per_attribute', 'dt_nodes_per_instance', 'lr_mean_accuracy', 'lr_std_accuracy']
        
        # Default landmarking features
        landmark_features = ['nb_landmark', '1nn_landmark', 'random_node_landmark', 
                           'worst_node_landmark', 'linear_discriminant_landmark']
        
        all_features = stat_features + info_features + model_features + landmark_features
        
        for feature in all_features:
            default_features[feature] = 0.0
            
        return default_features