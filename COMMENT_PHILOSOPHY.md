# 💡 Comment Philosophy Update

## What Changed

Updated documentation to reflect best practice: **Only comment when necessary**.

## The Principle

```
If function name is clear → No comment
If code is obvious → No comment
If logic is non-obvious → Comment the WHY
If preventing bugs → Mark as CRITICAL
```

## Examples

### ❌ DON'T
```python
# Load the dataset
dataset = load_openml_dataset(dataset_id)

# Get X and y
X, y = dataset['X'], dataset['y']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
Too many obvious comments = noise.

### ✅ DO
```python
dataset = load_openml_dataset(dataset_id)
X, y = dataset['X'], dataset['y']

# CRITICAL: Fit on train only to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
preprocessor.fit(X_train, y_train)
X_test = preprocessor.transform(X_test, y_test)
```
Comments only where they add value.

## Key Rule

**Good variable names + clear function names = fewer comments needed**

```python
# BAD - needs comment:
# Skip datasets where all features are constant
for d in dataset_ids:
    if len(set(X[d])) == 1:
        continue

# GOOD - self-explanatory:
for dataset_id in dataset_ids:
    if is_constant_feature(X[dataset_id]):
        continue
```

## Benefits

✅ Code is cleaner and less cluttered  
✅ Comments stand out when they appear  
✅ Easier to maintain (fewer comments to update)  
✅ Forces better naming conventions  
✅ Professional/production-grade style  

---

**Updated**: October 18, 2025  
**Files Updated**: COMMENTS_GUIDE.md, README.md
