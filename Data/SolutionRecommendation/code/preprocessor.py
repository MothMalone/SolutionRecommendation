class Preprocessor:
    """A stateful class to handle fitting and transforming data to prevent data leakage."""
    def __init__(self, config):
        self.config = config
        self.column_transformer = None
        self.selection_model = None
        self.reduction_model = None
        self.outlier_models = {}
        self.fitted_columns = None
        self.fitted = False

    def fit(self, X, y):
        try:
            X_processed = X.copy()
            y_processed = y.copy()
            
            #  Fit ColumnTransformer for imputation, scaling, encoding 
            numeric_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
            
            imputers = {
                'mean': SimpleImputer(strategy='mean'), 
                'median': SimpleImputer(strategy='median'), 
                'knn': KNNImputer(n_neighbors=min(5, len(X)-1)), 
                'most_frequent': SimpleImputer(strategy='most_frequent'), 
                'constant': SimpleImputer(strategy='constant', fill_value=0)
            }
            scalers = {
                'standard': StandardScaler(), 
                'minmax': MinMaxScaler(), 
                'robust': RobustScaler(), 
                'maxabs': MaxAbsScaler()
            }
            
            numeric_steps = []
            if self.config.get('imputation') in imputers: 
                numeric_steps.append(('imputer', imputers[self.config['imputation']]))
            if self.config.get('scaling') in scalers: 
                numeric_steps.append(('scaler', scalers[self.config['scaling']]))
            numeric_pipeline = Pipeline(steps=numeric_steps) if numeric_steps else 'passthrough'

            categorical_pipeline = 'passthrough'
            if self.config.get('encoding') == 'onehot' and categorical_features:
                categorical_pipeline = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')), 
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=50))
                ])

            transformers = []
            if numeric_features:
                transformers.append(('num', numeric_pipeline, numeric_features))
            if categorical_features:
                transformers.append(('cat', categorical_pipeline, categorical_features))
            
            if not transformers:
                raise ValueError("No features to transform")
                
            self.column_transformer = ColumnTransformer(transformers, remainder='drop')
            self.column_transformer.fit(X_processed)
            X_processed = pd.DataFrame(
                self.column_transformer.transform(X_processed), 
                columns=self.column_transformer.get_feature_names_out(), 
                index=X.index
            )
            
            if X_processed.shape[1] == 0:
                raise ValueError("No features remained after initial transformation")
            
            # Fit Feature Selection Models
            if self.config['feature_selection'] in ['k_best', 'mutual_info']:
                k = min(20, X_processed.shape[1])
                if k > 0 and len(X_processed) > k:  # Need enough samples
                    selector_func = f_classif if self.config['feature_selection'] == 'k_best' else mutual_info_classif
                    self.selection_model = SelectKBest(selector_func, k=k)
                    self.selection_model.fit(X_processed, y_processed)
                    X_processed = pd.DataFrame(
                        self.selection_model.transform(X_processed), 
                        columns=self.selection_model.get_feature_names_out(), 
                        index=X_processed.index
                    )
            elif self.config['feature_selection'] == 'variance_threshold':
                if X_processed.shape[1] > 1:
                    selector = VarianceThreshold(threshold=0.0)
                    selector.fit(X_processed)
                    if selector.transform(X_processed).shape[1] > 0:
                        self.selection_model = selector
                        X_processed = pd.DataFrame(
                            selector.transform(X_processed),
                            index=X_processed.index
                        )
            
            # Fit Dimensionality Reduction Models 
            if self.config['dimensionality_reduction'] in ['pca', 'svd']:
                n_components = min(10, X_processed.shape[1], len(X_processed) - 1)
                if n_components > 0:
                    reducer = PCA(n_components=n_components) if self.config['dimensionality_reduction'] == 'pca' else TruncatedSVD(n_components=n_components)
                    self.reduction_model = reducer.fit(X_processed)

            # Fit Outlier Removal Models
            if self.config['outlier_removal'] in ['isolation_forest', 'lof'] and len(X_processed) > 10:
                model = IsolationForest(random_state=42, contamination=0.05) if self.config['outlier_removal'] == 'isolation_forest' else LocalOutlierFactor()
                self.outlier_models['model'] = model.fit(X_processed)

            self.fitted = True
            self.fitted_columns = X_processed.columns
            
        except Exception as e:
            print(f"      Warning: Preprocessor fit failed: {e}")
            self.fitted = False

    def transform(self, X, y):
        if not self.fitted: 
            print("      Warning: Preprocessor not fitted, returning original data")
            return X.reset_index(drop=True), y.reset_index(drop=True)
        
        try:
            X_processed = X.copy()
            y_processed = y.copy()

            # Impute, scale, encode
            X_processed = pd.DataFrame(
                self.column_transformer.transform(X_processed), 
                columns=self.column_transformer.get_feature_names_out(), 
                index=X_processed.index
            )
            
            # Feature selection
            if self.selection_model:
                X_processed = pd.DataFrame(
                    self.selection_model.transform(X_processed), 
                    columns=self.selection_model.get_feature_names_out() if hasattr(self.selection_model, 'get_feature_names_out') else None,
                    index=X_processed.index
                )
            
            # Dimensionality reduction
            if self.reduction_model:
                X_processed = pd.DataFrame(
                    self.reduction_model.transform(X_processed), 
                    index=X_processed.index
                )
            
            if self.config['outlier_removal'] != 'none' and len(X_processed) > 20:
                original_size = len(X_processed)
                
                if self.config['outlier_removal'] == 'iqr':
                    mask = pd.Series(True, index=X_processed.index)
                    for col in X_processed.columns:
                        q1, q3 = X_processed[col].quantile(0.25), X_processed[col].quantile(0.75)
                        iqr = q3 - q1
                        if iqr > 0:  # Avoid division by zero
                            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                            mask = mask & (X_processed[col] >= lower) & (X_processed[col] <= upper)
                    X_processed = X_processed[mask]
                    
                elif self.config['outlier_removal'] == 'zscore':
                    try:
                        z_scores = np.abs(zscore(X_processed, nan_policy='omit'))
                        mask = (z_scores < 3).all(axis=1)
                        X_processed = X_processed[mask]
                    except:
                        pass  # Keep original data if zscore fails
                        
                elif self.config['outlier_removal'] in ['isolation_forest', 'lof'] and 'model' in self.outlier_models:
                    try:
                        predictions = self.outlier_models['model'].fit_predict(X_processed)
                        X_processed = X_processed[predictions == 1]
                    except:
                        pass  # Keep original data if outlier detection fails
                
                if len(X_processed) < original_size * 0.1:  # Keep at least 10% of data
                    print(f"      Warning: Outlier removal would remove too much data, skipping")
                    X_processed = X.copy()
                    X_processed = pd.DataFrame(
                        self.column_transformer.transform(X_processed), 
                        columns=self.column_transformer.get_feature_names_out(), 
                        index=X_processed.index
                    )
                    if self.selection_model:
                        X_processed = pd.DataFrame(
                            self.selection_model.transform(X_processed), 
                            index=X_processed.index
                        )
                    if self.reduction_model:
                        X_processed = pd.DataFrame(
                            self.reduction_model.transform(X_processed), 
                            index=X_processed.index
                        )
                else:
                    y_processed = y_processed.loc[X_processed.index]

            return X_processed.reset_index(drop=True), y_processed.reset_index(drop=True)
            
        except Exception as e:
            print(f"      Warning: Transform failed: {e}, returning original data")
            return X.reset_index(drop=True), y.reset_index(drop=True)