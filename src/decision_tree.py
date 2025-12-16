import numpy as np
from typing import Optional, Union, List, Tuple


class Node:
    
    def __init__(
        self,
        is_leaf: bool = False,
        prediction: Optional[float] = None,
        feature_index: Optional[int] = None,
        feature_name: Optional[str] = None,
        is_categorical: bool = False,
        threshold: Optional[float] = None,
        categories_left: Optional[set] = None,
        categories_right: Optional[set] = None,
        left: Optional['Node'] = None,
        right: Optional['Node'] = None,
        n_samples: int = 0,
        mse: float = 0.0,
        depth: int = 0
    ):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature_index = feature_index
        self.feature_name = feature_name
        self.is_categorical = is_categorical
        self.threshold = threshold
        self.categories_left = categories_left if categories_left is not None else set()
        self.categories_right = categories_right if categories_right is not None else set()
        self.left = left
        self.right = right
        self.n_samples = n_samples
        self.mse = mse
        self.depth = depth
    
    def __repr__(self) -> str:
        if self.is_leaf:
            return f"Leaf(pred={self.prediction:.4f}, n={self.n_samples})"
        elif self.is_categorical:
            return (f"Node(feat={self.feature_name}, categorical, "
                    f"left={self.categories_left}, n={self.n_samples})")
        else:
            return (f"Node(feat={self.feature_name}, "
                    f"thresh={self.threshold:.4f}, n={self.n_samples})")

# Decision Tree Regressor with native categorical feature support
class DecisionTreeRegressor:
    
    # Optimized implementation using:
    # - Sorted arrays for numerical splits (O(n log n) sort + O(unique) scan)
    # - Mean-based ordering for categorical splits (optimal for MSE criterion)
    
    # Parameters
    # - max_depth: Maximum depth of the tree. None for unlimited depth.
    # - min_samples_split: Minimum samples required to split an internal node.
    # - min_samples_leaf: Minimum samples required in a leaf node.
    # - random_state: Random seed for reproducibility.
    # - verbose: Whether to print progress messages.
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
        verbose: bool = True
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.verbose = verbose
        
        # Tree structure
        self.root: Optional[Node] = None
        
        # Training metadata
        self.n_features_: int = 0
        self.feature_names_: List[str] = []
        self.feature_types_: dict = {}
        self.n_samples_: int = 0
        
        # Feature importance (variance reduction weighted by samples)
        self.feature_importances_: Optional[np.ndarray] = None
        self._feature_importance_accumulator: dict = {}
        
        # Split statistics for diagnostics
        self._split_counts: dict = {}
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def __repr__(self) -> str:
        if self.root is None:
            return "DecisionTreeRegressor(not fitted)"
        return (f"DecisionTreeRegressor(max_depth={self.max_depth}, "
                f"n_samples={self.n_samples_}, n_features={self.n_features_})")
    
    def _calculate_mse(self, y: np.ndarray) -> float:
        # Calculate MSE (variance) of target array.
        if len(y) == 0:
            return 0.0
        return np.var(y)
    
    def _calculate_variance_reduction(
        self, 
        y_parent: np.ndarray, 
        y_left: np.ndarray, 
        y_right: np.ndarray
    ) -> float:
        # Calculate variance reduction from a split.
        if len(y_left) == 0 or len(y_right) == 0:
            return 0.0
        
        n = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)
        
        var_parent = np.var(y_parent)
        var_left = np.var(y_left)
        var_right = np.var(y_right)
        
        weighted_child_var = (n_left / n) * var_left + (n_right / n) * var_right
        return var_parent - weighted_child_var
    
    def _find_best_numerical_split_optimized(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_index: int
    ) -> Tuple[Optional[float], float]:
        # Find best numerical split using sorted arrays.
        # Uses cumulative sum approach for O(n log n) complexity instead of
        # O(n * unique_values) brute force.
        feature_values = X[:, feature_index].astype(float)  # Ensure numeric type
        n = len(y)
        
        # Sort by feature value
        sort_idx = np.argsort(feature_values)
        sorted_values = feature_values[sort_idx]
        sorted_y = y[sort_idx]
        
        # Find unique value boundaries
        unique_values, first_indices = np.unique(sorted_values, return_index=True)
        
        if len(unique_values) < 2:
            return None, 0.0
        
        # Cumulative sums for efficient MSE calculation
        cumsum_y = np.cumsum(sorted_y)
        cumsum_y2 = np.cumsum(sorted_y ** 2)
        
        total_sum = cumsum_y[-1]
        total_sum2 = cumsum_y2[-1]
        
        best_threshold = None
        best_reduction = 0.0
        var_parent = np.var(y)
        
        # Evaluate splits at boundaries between unique values
        for i in range(len(unique_values) - 1):
            split_point = first_indices[i + 1]
            n_left = split_point
            n_right = n - n_left
            
            # Check min_samples_leaf constraint
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue
            
            # Calculate left and right statistics using cumulative sums
            sum_left = cumsum_y[split_point - 1]
            sum2_left = cumsum_y2[split_point - 1]
            
            sum_right = total_sum - sum_left
            sum2_right = total_sum2 - sum2_left
            
            # Variance = E[X^2] - E[X]^2
            var_left = sum2_left / n_left - (sum_left / n_left) ** 2
            var_right = sum2_right / n_right - (sum_right / n_right) ** 2
            
            # Variance reduction
            weighted_child_var = (n_left / n) * var_left + (n_right / n) * var_right
            reduction = var_parent - weighted_child_var
            
            if reduction > best_reduction:
                best_reduction = reduction
                # Threshold is midpoint between adjacent unique values
                best_threshold = (unique_values[i] + unique_values[i + 1]) / 2
        
        return best_threshold, best_reduction
    
    def _find_best_categorical_split_optimized(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_index: int
    ) -> Tuple[Optional[set], Optional[set], float]:
        # Find best categorical split using mean-based ordering.
        # For MSE criterion, the optimal binary partition of categories can be found
        # by sorting categories by their mean target value and finding the best
        # split point. This is O(k log k) where k is number of categories.
        feature_values = X[:, feature_index]
        unique_categories = np.unique(feature_values)
        
        if len(unique_categories) < 2:
            return None, None, 0.0
        
        # Calculate mean target for each category
        category_stats = []
        for cat in unique_categories:
            mask = feature_values == cat
            cat_y = y[mask]
            if len(cat_y) > 0:
                category_stats.append({
                    'category': cat,
                    'mean': np.mean(cat_y),
                    'count': len(cat_y),
                    'sum': np.sum(cat_y),
                    'sum2': np.sum(cat_y ** 2)
                })
        
        if len(category_stats) < 2:
            return None, None, 0.0
        
        # Sort categories by mean target value
        category_stats.sort(key=lambda x: x['mean'])
        
        n = len(y)
        var_parent = np.var(y)
        
        best_categories_left = None
        best_categories_right = None
        best_reduction = 0.0
        
        # Try all possible splits along the sorted order
        # This guarantees finding the optimal binary split for MSE
        cumsum = 0.0
        cumsum2 = 0.0
        cumcount = 0
        
        total_sum = sum(s['sum'] for s in category_stats)
        total_sum2 = sum(s['sum2'] for s in category_stats)
        
        for i in range(len(category_stats) - 1):
            stat = category_stats[i]
            cumsum += stat['sum']
            cumsum2 += stat['sum2']
            cumcount += stat['count']
            
            n_left = cumcount
            n_right = n - n_left
            
            # Check min_samples_leaf constraint
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue
            
            sum_right = total_sum - cumsum
            sum2_right = total_sum2 - cumsum2
            
            # Variance = E[X^2] - E[X]^2
            var_left = cumsum2 / n_left - (cumsum / n_left) ** 2
            var_right = sum2_right / n_right - (sum_right / n_right) ** 2
            
            # Variance reduction
            weighted_child_var = (n_left / n) * var_left + (n_right / n) * var_right
            reduction = var_parent - weighted_child_var
            
            if reduction > best_reduction:
                best_reduction = reduction
                best_categories_left = {s['category'] for s in category_stats[:i+1]}
                best_categories_right = {s['category'] for s in category_stats[i+1:]}
        
        return best_categories_left, best_categories_right, best_reduction
    
    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[Optional[int], Optional[Union[float, set]], Optional[set], bool, float]:
        # Find the best split across all features.
        best_feature_index = None
        best_threshold_or_cats_left = None
        best_categories_right = None
        best_is_categorical = False
        best_overall_reduction = 0.0
        
        for feature_index in range(self.n_features_):
            is_categorical = self.feature_types_.get(feature_index) == 'categorical'
            
            if is_categorical:
                cats_left, cats_right, reduction = self._find_best_categorical_split_optimized(
                    X, y, feature_index
                )
                if reduction > best_overall_reduction:
                    best_overall_reduction = reduction
                    best_feature_index = feature_index
                    best_threshold_or_cats_left = cats_left
                    best_categories_right = cats_right
                    best_is_categorical = True
            else:
                threshold, reduction = self._find_best_numerical_split_optimized(
                    X, y, feature_index
                )
                if reduction > best_overall_reduction:
                    best_overall_reduction = reduction
                    best_feature_index = feature_index
                    best_threshold_or_cats_left = threshold
                    best_categories_right = None
                    best_is_categorical = False
        
        return (
            best_feature_index,
            best_threshold_or_cats_left,
            best_categories_right,
            best_is_categorical,
            best_overall_reduction
        )
    
    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int = 0
    ) -> Node:
        # Recursively build the decision tree.
        n_samples = len(y)
        current_mse = self._calculate_mse(y)
        prediction = np.mean(y)
        
        # Stopping criterion 1: max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(
                is_leaf=True,
                prediction=prediction,
                n_samples=n_samples,
                mse=current_mse,
                depth=depth
            )
        
        # Stopping criterion 2: too few samples to split
        if n_samples < self.min_samples_split:
            return Node(
                is_leaf=True,
                prediction=prediction,
                n_samples=n_samples,
                mse=current_mse,
                depth=depth
            )
        
        # Stopping criterion 3: pure node (all targets identical)
        if current_mse == 0:
            return Node(
                is_leaf=True,
                prediction=prediction,
                n_samples=n_samples,
                mse=current_mse,
                depth=depth
            )
        
        # Find best split
        (
            best_feature_idx,
            threshold_or_cats_left,
            cats_right,
            is_categorical,
            variance_reduction
        ) = self._find_best_split(X, y)
        
        # Stopping criterion 4: no valid split found
        if variance_reduction <= 0 or best_feature_idx is None:
            return Node(
                is_leaf=True,
                prediction=prediction,
                n_samples=n_samples,
                mse=current_mse,
                depth=depth
            )
        
        # Track feature importance and split counts
        feature_name = self.feature_names_[best_feature_idx]
        if feature_name not in self._feature_importance_accumulator:
            self._feature_importance_accumulator[feature_name] = 0.0
            self._split_counts[feature_name] = 0
        
        self._feature_importance_accumulator[feature_name] += variance_reduction * n_samples
        self._split_counts[feature_name] += 1
        
        # Create internal node and split data
        if is_categorical:
            feature_values = X[:, best_feature_idx]
            left_mask = np.array([val in threshold_or_cats_left for val in feature_values])
            right_mask = ~left_mask
            
            node = Node(
                is_leaf=False,
                feature_index=best_feature_idx,
                feature_name=feature_name,
                is_categorical=True,
                categories_left=threshold_or_cats_left,
                categories_right=cats_right,
                n_samples=n_samples,
                mse=current_mse,
                depth=depth
            )
        else:
            # Convert to float for numerical comparison
            feature_values = X[:, best_feature_idx].astype(float)
            threshold = threshold_or_cats_left
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            
            node = Node(
                is_leaf=False,
                feature_index=best_feature_idx,
                feature_name=feature_name,
                is_categorical=False,
                threshold=threshold,
                n_samples=n_samples,
                mse=current_mse,
                depth=depth
            )
        
        # Recursively build subtrees
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        node.left = self._build_tree(X_left, y_left, depth + 1)
        node.right = self._build_tree(X_right, y_right, depth + 1)
        
        return node
    
    def fit(
        self,
        X,
        y,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[Union[int, str]]] = None
    ) -> 'DecisionTreeRegressor':
        # Fit the decision tree regressor.
        # - X: Training features. Can be DataFrame with mixed types.
        # - y: Target values.
        # - feature_names: Feature names. Inferred from DataFrame columns if not provided.
        # - categorical_features: Indices or names of categorical features.
        # - Returns: Fitted estimator.
        # Handle DataFrame input
        if hasattr(X, 'columns'):
            if feature_names is None:
                feature_names = list(X.columns)
            X = X.values
        
        if hasattr(y, 'values'):
            y = y.values
        
        X = np.asarray(X)
        y = np.asarray(y).astype(float)
        
        # Validate input
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}"
            )
        
        # Store metadata
        self.n_samples_ = X.shape[0]
        self.n_features_ = X.shape[1]
        
        # Handle feature names
        if feature_names is None:
            self.feature_names_ = [f"feature_{i}" for i in range(self.n_features_)]
        else:
            if len(feature_names) != self.n_features_:
                raise ValueError(
                    f"Number of feature_names ({len(feature_names)}) must match "
                    f"number of features ({self.n_features_})"
                )
            self.feature_names_ = list(feature_names)
        
        # Determine feature types
        self.feature_types_ = {i: 'numerical' for i in range(self.n_features_)}
        if categorical_features is not None:
            for cat_feature in categorical_features:
                if isinstance(cat_feature, str):
                    if cat_feature not in self.feature_names_:
                        raise ValueError(f"Feature '{cat_feature}' not found in feature_names")
                    feature_idx = self.feature_names_.index(cat_feature)
                else:
                    feature_idx = cat_feature
                    if feature_idx < 0 or feature_idx >= self.n_features_:
                        raise ValueError(
                            f"Feature index {feature_idx} out of range [0, {self.n_features_-1}]"
                        )
                self.feature_types_[feature_idx] = 'categorical'
        
        # Reset tracking
        self._feature_importance_accumulator = {}
        self._split_counts = {}
        
        # Build tree
        cat_feature_names = [
            self.feature_names_[i] 
            for i, t in self.feature_types_.items() 
            if t == 'categorical'
        ]
        if self.verbose:
            print(f"Building tree with {self.n_samples_:,} samples and {self.n_features_} features...")
            print(f"Categorical features: {cat_feature_names}")
        
        self.root = self._build_tree(X, y, depth=0)
        
        # Compute normalized feature importances
        total_importance = sum(self._feature_importance_accumulator.values())
        if total_importance > 0:
            self.feature_importances_ = np.zeros(self.n_features_)
            for feat_name, importance in self._feature_importance_accumulator.items():
                idx = self.feature_names_.index(feat_name)
                self.feature_importances_[idx] = importance / total_importance
        
        if self.verbose:
            print(f"Tree built successfully!")
        return self
    
    def _predict_single(self, x: np.ndarray, node: Node) -> float:
        # Predict for a single sample.
        if node.is_leaf:
            return node.prediction
        
        feature_value = x[node.feature_index]
        
        if node.is_categorical:
            if feature_value in node.categories_left:
                return self._predict_single(x, node.left)
            else:
                return self._predict_single(x, node.right)
        else:
            # Convert to float for numerical comparison
            if float(feature_value) <= node.threshold:
                return self._predict_single(x, node.left)
            else:
                return self._predict_single(x, node.right)
    
    def predict(self, X) -> np.ndarray:
        
        # Predict target values for samples in X.
        # - X: array-like of shape (n_samples, n_features)
        # - Returns: ndarray of shape (n_samples,)
        # - Predicted values.
        
        if self.root is None:
            raise ValueError("Tree has not been fitted yet. Call fit() before predict().")
        
        if hasattr(X, 'values'):
            X = X.values
        
        X = np.asarray(X)
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but tree was trained with "
                f"{self.n_features_} features"
            )
        
        predictions = np.array([self._predict_single(x, self.root) for x in X])
        return predictions
    
    def get_n_leaves(self) -> int:
        # Get the number of leaf nodes in the tree.
        if self.root is None:
            return 0
        
        def count_leaves(node: Node) -> int:
            if node.is_leaf:
                return 1
            return count_leaves(node.left) + count_leaves(node.right)
        
        return count_leaves(self.root)
    
    def get_depth(self) -> int:
        # Get the actual depth of the tree.
        if self.root is None:
            return 0
        
        def get_max_depth(node: Node) -> int:
            if node.is_leaf:
                return node.depth
            return max(get_max_depth(node.left), get_max_depth(node.right))
        
        return get_max_depth(self.root)
    
    def get_split_summary(self) -> dict:
        
        # Get a summary of splits by feature type.
        # Returns a dict with split counts per feature and per feature type.
        summary = {
            'by_feature': dict(self._split_counts),
            'numerical_splits': 0,
            'categorical_splits': 0,
            'feature_importances': {}
        }
        
        for feat_name, count in self._split_counts.items():
            idx = self.feature_names_.index(feat_name)
            is_cat = self.feature_types_[idx] == 'categorical'
            
            if is_cat:
                summary['categorical_splits'] += count
            else:
                summary['numerical_splits'] += count
        
        if self.feature_importances_ is not None:
            for i, imp in enumerate(self.feature_importances_):
                if imp > 0:
                    summary['feature_importances'][self.feature_names_[i]] = imp
        
        return summary
    
    def print_tree(self, max_depth: Optional[int] = None) -> str:
        
        # Return a string representation of the tree structure.
        
        # - max_depth: Maximum depth to display. None shows full tree.
        # - Returns: Tree structure as string
        
        if self.root is None:
            return "Tree not fitted"
        
        lines = []
        
        def print_node(node: Node, prefix: str = "", is_left: bool = True):
            if max_depth is not None and node.depth > max_depth:
                return
            
            connector = "├── " if is_left else "└── "
            extension = "│   " if is_left else "    "
            
            if node.depth == 0:
                lines.append(f"Root: {node}")
            else:
                side = "Left" if is_left else "Right"
                lines.append(f"{prefix}{connector}[{side}] {node}")
            
            if not node.is_leaf:
                new_prefix = prefix + extension if node.depth > 0 else ""
                print_node(node.left, new_prefix, is_left=True)
                print_node(node.right, new_prefix, is_left=False)
        
        print_node(self.root)
        return "\n".join(lines)
