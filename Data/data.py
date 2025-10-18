import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Dataset Preprocessing Inspector",
    page_icon="🔍",
    layout="wide"
)

# Dataset classifications
train_need_preprocessing = [22, 23, 24, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 39, 42, 43, 48, 49, 50, 53, 54, 55, 56, 59, 61, 62, 163, 164, 171, 181, 182, 185, 186, 188, 275, 276, 277, 278, 285, 300, 301, 307, 311, 312, 313, 316, 327, 328, 329, 333, 334, 335, 336, 337, 338, 339, 340, 342, 343, 346, 372, 375, 378, 443, 444, 446, 448, 450, 451, 452, 453, 454, 455, 457, 458, 459, 461, 462, 463, 465, 467, 469]
train_no_preprocessing = [40, 41, 60, 187, 308, 310, 464, 468]
test_need_preprocessing = [3, 5, 6, 8, 9, 10, 11, 12, 183, 255, 475, 481, 516, 546, 1503, 1551, 1552]
test_no_preprocessing = [14, 23517]

all_datasets = sorted(train_need_preprocessing + train_no_preprocessing + 
                     test_need_preprocessing + test_no_preprocessing)

@st.cache_data
def load_dataset(dataset_id):
    """Load and preprocess dataset from OpenML"""
    try:
        dataset = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
        X, y = dataset.data, dataset.target
        
        # Handle categorical columns
        if isinstance(X, pd.DataFrame):
            categorical_cols = X.select_dtypes(['category']).columns
            if len(categorical_cols) > 0:
                X = X.copy()
                X.loc[:, categorical_cols] = X.loc[:, categorical_cols].astype(object)
        
        # Handle target encoding
        if y.dtype == 'object' or y.dtype.name == 'category':
            y = pd.Series(LabelEncoder().fit_transform(y), name=y.name)
        
        # Remove rows with missing targets
        valid_indices = y.dropna().index
        X = X.loc[valid_indices].reset_index(drop=True)
        y = y.loc[valid_indices].reset_index(drop=True)
        
        metadata = {
            'id': dataset_id,
            'name': dataset.details.get('name', 'Unknown'),
            'n_samples': X.shape[0],
            'n_features': X.shape[1]
        }
        
        return X, y, metadata, None
    except Exception as e:
        return None, None, None, str(e)

def get_classification(dataset_id):
    """Get classification and color for dataset"""
    if dataset_id in train_need_preprocessing:
        return "🔴 TRAIN - NEEDS PREPROCESSING", "red"
    elif dataset_id in train_no_preprocessing:
        return "🟢 TRAIN - NO PREPROCESSING", "green"
    elif dataset_id in test_need_preprocessing:
        return "🟠 TEST - NEEDS PREPROCESSING", "orange"
    else:
        return "🔵 TEST - NO PREPROCESSING", "blue"

def detect_custom_missing(series):
    """Detect custom missing value indicators"""
    if series.dtype == 'object':
        suspicious = series.isin(['?', 'NA', 'N/A', 'null', 'NULL', '', ' ', '...', 'nan', 'NaN']).sum()
        return suspicious
    return 0

def analyze_categorical_potential(X):
    """Identify numeric columns that might be categorical"""
    potential_categorical = []
    for col in X.select_dtypes(include=np.number).columns:
        n_unique = X[col].nunique()
        if n_unique <= 20:
            potential_categorical.append({
                'column': col,
                'n_unique': n_unique,
                'values': sorted(X[col].unique().tolist())
            })
    return potential_categorical

def analyze_high_correlation(X):
    """Find highly correlated feature pairs"""
    numeric_cols = X.select_dtypes(include=np.number).columns
    high_corr = []
    
    if len(numeric_cols) > 1:
        try:
            corr_matrix = X[numeric_cols].corr().abs()
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.8:
                        high_corr.append({
                            'col1': corr_matrix.columns[i],
                            'col2': corr_matrix.columns[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })
        except:
            pass
    
    return sorted(high_corr, key=lambda x: x['correlation'], reverse=True)

# Title
st.title("🔍 Dataset Preprocessing Inspector")
st.markdown("---")

# Sidebar navigation
with st.sidebar:
    st.header("Navigation")
    
    # Dataset selector
    current_idx = st.number_input(
        "Dataset Index",
        min_value=1,
        max_value=len(all_datasets),
        value=1,
        step=1
    ) - 1
    
    dataset_id = all_datasets[current_idx]
    
    st.markdown(f"**Dataset ID:** `{dataset_id}`")
    st.markdown(f"**Position:** {current_idx + 1} / {len(all_datasets)}")
    
    # Quick navigation
    col1, col2 = st.columns(2)
    if col1.button("⬅️ Previous"):
        if current_idx > 0:
            st.session_state.dataset_idx = current_idx - 1
            st.rerun()
    
    if col2.button("Next ➡️"):
        if current_idx < len(all_datasets) - 1:
            st.session_state.dataset_idx = current_idx + 1
            st.rerun()
    
    st.markdown("---")
    
    # Direct selection
    st.subheader("Jump to Dataset")
    selected_id = st.selectbox(
        "Select Dataset ID",
        all_datasets,
        index=current_idx
    )
    
    if selected_id != dataset_id:
        new_idx = all_datasets.index(selected_id)
        st.session_state.dataset_idx = new_idx
        st.rerun()
    
    st.markdown("---")
    
    # Legend
    st.subheader("Classification Legend")
    st.markdown("🔴 Train - Needs Preprocessing")
    st.markdown("🟢 Train - No Preprocessing")
    st.markdown("🟠 Test - Needs Preprocessing")
    st.markdown("🔵 Test - No Preprocessing")

# Main content
classification, color = get_classification(dataset_id)
st.markdown(f"## {classification}")

# Load dataset
with st.spinner(f"Loading dataset {dataset_id}..."):
    X, y, metadata, error = load_dataset(dataset_id)

if error:
    st.error(f"Failed to load dataset: {error}")
    st.stop()

# Display basic info
col1, col2, col3, col4 = st.columns(4)
col1.metric("Dataset ID", metadata['id'])
col2.metric("Samples", f"{metadata['n_samples']:,}")
col3.metric("Features", metadata['n_features'])
col4.metric("Dataset Name", metadata['name'][:20] + "..." if len(metadata['name']) > 20 else metadata['name'])

# Tabs for different views
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", 
    "🔢 Data Preview", 
    "❓ Missing Values", 
    "🏷️ Categorical Analysis",
    "📈 Statistics",
    "🔗 Correlations"
])

# TAB 1: Overview
with tab1:
    st.subheader("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Feature Types**")
        numeric_cols = X.select_dtypes(include=np.number).columns
        object_cols = X.select_dtypes(exclude=np.number).columns
        
        st.write(f"- Numeric Features: {len(numeric_cols)}")
        st.write(f"- Object/String Features: {len(object_cols)}")
        
        st.markdown("**Missing Values**")
        total_missing = X.isnull().sum().sum()
        total_cells = X.shape[0] * X.shape[1]
        st.write(f"- Total Missing: {total_missing:,}")
        st.write(f"- Percentage: {(total_missing / total_cells) * 100:.4f}%")
    
    with col2:
        st.markdown("**Potential Issues**")
        
        # Constant features
        constant_cols = [col for col in X.columns if X[col].nunique() == 1]
        if constant_cols:
            st.warning(f"⚠️ {len(constant_cols)} constant feature(s)")
        
        # High cardinality categorical
        high_card = []
        for col in object_cols:
            if X[col].nunique() > len(X) * 0.5:
                high_card.append(col)
        if high_card:
            st.warning(f"⚠️ {len(high_card)} high cardinality categorical feature(s)")
        
        # Low cardinality numeric
        potential_cat = analyze_categorical_potential(X)
        if potential_cat:
            st.warning(f"⚠️ {len(potential_cat)} numeric feature(s) might be categorical")
        
        # High correlation
        high_corr = analyze_high_correlation(X)
        if high_corr:
            st.warning(f"⚠️ {len(high_corr)} highly correlated pair(s)")
    
    # Feature lists
    st.markdown("---")
    st.markdown("**Feature Names**")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.expander(f"Numeric Columns ({len(numeric_cols)})"):
            st.write(list(numeric_cols))
    
    with col2:
        with st.expander(f"Object/String Columns ({len(object_cols)})"):
            st.write(list(object_cols))

# TAB 2: Data Preview
with tab2:
    st.subheader("Data Preview (First 100 rows)")
    
    # Show first 100 rows
    st.dataframe(X.head(100), use_container_width=True, height=400)
    
    st.markdown("---")
    st.markdown("**Column Data Types**")
    dtype_df = pd.DataFrame({
        'Column': X.columns,
        'Type': X.dtypes.astype(str),
        'Non-Null Count': X.count(),
        'Null Count': X.isnull().sum(),
        'Unique Values': [X[col].nunique() for col in X.columns]
    })
    st.dataframe(dtype_df, use_container_width=True)

# TAB 3: Missing Values
with tab3:
    st.subheader("Missing Values Analysis")
    
    missing_cols = X.isnull().sum()
    missing_cols = missing_cols[missing_cols > 0].sort_values(ascending=False)
    
    if len(missing_cols) > 0:
        st.markdown(f"**Found {len(missing_cols)} column(s) with missing values**")
        
        missing_df = pd.DataFrame({
            'Column': missing_cols.index,
            'Missing Count': missing_cols.values,
            'Missing %': (missing_cols.values / len(X) * 100).round(2),
            'Data Type': [X[col].dtype for col in missing_cols.index],
            'Non-Missing Unique': [X[col].nunique() for col in missing_cols.index]
        })
        
        st.dataframe(missing_df, use_container_width=True)
        
        # Detailed view for each column
        st.markdown("---")
        st.markdown("**Detailed Analysis by Column**")
        
        for col in missing_cols.index[:10]:  # Show first 10
            with st.expander(f"{col} - {missing_cols[col]:,} missing ({(missing_cols[col] / len(X) * 100):.2f}%)"):
                st.write(f"**Data Type:** {X[col].dtype}")
                st.write(f"**Non-null Unique Values:** {X[col].nunique()}")
                
                st.write("**Top 10 Values:**")
                st.dataframe(X[col].value_counts().head(10))
    else:
        st.success("✅ No missing values detected!")
    
    # Custom missing indicators
    st.markdown("---")
    st.subheader("Custom Missing Value Indicators")
    
    custom_missing = []
    for col in object_cols:
        count = detect_custom_missing(X[col])
        if count > 0:
            custom_missing.append({'Column': col, 'Count': count})
    
    if custom_missing:
        st.warning(f"⚠️ Found custom missing indicators in {len(custom_missing)} column(s)")
        st.dataframe(pd.DataFrame(custom_missing), use_container_width=True)
    else:
        st.success("✅ No custom missing indicators detected")

# TAB 4: Categorical Analysis
with tab4:
    st.subheader("Categorical Features Analysis")
    
    # Explicit categorical
    object_cols = X.select_dtypes(exclude=np.number).columns
    
    if len(object_cols) > 0:
        st.markdown(f"**Explicit Categorical Features: {len(object_cols)}**")
        
        for col in object_cols:
            n_unique = X[col].nunique()
            n_total = len(X.dropna(subset=[col]))
            
            with st.expander(f"{col} - {n_unique} unique values (cardinality: {n_unique/n_total:.4f})"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write(f"**Unique Values:** {n_unique}")
                    st.write(f"**Missing:** {X[col].isnull().sum()}")
                    st.write(f"**Cardinality Ratio:** {n_unique/n_total:.4f}")
                
                with col2:
                    st.write("**Top 10 Values:**")
                    value_counts = X[col].value_counts().head(10)
                    value_df = pd.DataFrame({
                        'Value': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': (value_counts.values / n_total * 100).round(2)
                    })
                    st.dataframe(value_df, use_container_width=True)
    else:
        st.info("No explicit categorical features found")
    
    # Potentially categorical numeric
    st.markdown("---")
    st.subheader("Potentially Categorical Numeric Features")
    
    potential_cat = analyze_categorical_potential(X)
    
    if potential_cat:
        st.warning(f"⚠️ Found {len(potential_cat)} numeric feature(s) with low cardinality (≤20 unique values)")
        
        for item in potential_cat:
            with st.expander(f"{item['column']} - {item['n_unique']} unique values"):
                st.write(f"**Unique Values:** {item['values']}")
                
                value_counts = X[item['column']].value_counts().sort_index()
                value_df = pd.DataFrame({
                    'Value': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': (value_counts.values / len(X) * 100).round(2)
                })
                st.dataframe(value_df, use_container_width=True)
    else:
        st.success("✅ All numeric features have appropriate cardinality")

# TAB 5: Statistics
with tab5:
    st.subheader("Descriptive Statistics")
    
    numeric_cols = X.select_dtypes(include=np.number).columns
    
    if len(numeric_cols) > 0:
        st.dataframe(X[numeric_cols].describe(), use_container_width=True)
        
        st.markdown("---")
        st.markdown("**Distribution Checks**")
        
        # Check for constant/quasi-constant
        constant = []
        quasi_constant = []
        
        for col in numeric_cols:
            if X[col].nunique() == 1:
                constant.append(col)
            else:
                value_counts = X[col].value_counts(normalize=True)
                if value_counts.iloc[0] > 0.95:
                    quasi_constant.append({'Column': col, 'Dominant %': f"{value_counts.iloc[0] * 100:.2f}%"})
        
        if constant:
            st.warning(f"⚠️ Constant features: {', '.join(constant)}")
        
        if quasi_constant:
            st.warning("⚠️ Quasi-constant features (>95% same value):")
            st.dataframe(pd.DataFrame(quasi_constant), use_container_width=True)
        
        if not constant and not quasi_constant:
            st.success("✅ No constant or quasi-constant features")
    else:
        st.info("No numeric features to analyze")

# TAB 6: Correlations
with tab6:
    st.subheader("Feature Correlations")
    
    numeric_cols = X.select_dtypes(include=np.number).columns
    
    if len(numeric_cols) > 1:
        high_corr = analyze_high_correlation(X)
        
        if high_corr:
            st.warning(f"⚠️ Found {len(high_corr)} highly correlated pair(s) (|r| > 0.8)")
            
            corr_df = pd.DataFrame(high_corr)
            st.dataframe(corr_df, use_container_width=True)
        else:
            st.success("✅ No highly correlated features (|r| > 0.8)")
        
        st.markdown("---")
        st.markdown("**Full Correlation Matrix**")
        
        # Show correlation heatmap data
        corr_matrix = X[numeric_cols].corr()
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1), 
                    use_container_width=True)
    else:
        st.info("Not enough numeric features for correlation analysis")

# Footer
st.markdown("---")
st.markdown(f"*Dataset {dataset_id} - {current_idx + 1} of {len(all_datasets)}*")