import os
import tempfile
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import (
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score,
    # Regression metrics
    r2_score, mean_squared_error, mean_absolute_error, 
    mean_absolute_percentage_error
)
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = tempfile.gettempdir()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def suggest_targets(df):
    suggestions = []
    for col in df.columns:
        unique = df[col].nunique()
        suggested = (unique <= 20 and unique > 1)
        suggestions.append({
            "column": col,
            "unique_values": int(unique),
            "suggested": suggested
        })
    return suggestions

def handle_missing_values(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        else:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def handle_outliers(df, outlier_cols=None):
    df = df.copy()
    if outlier_cols is None:
        outlier_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    
    return df

def encode_features(X):
    X = X.copy()
    encoders = {}
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    return X, encoders

def encode_target(y):
    if y.dtype == 'object' or str(y.dtype).startswith('category'):
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        return y, le
    return y, None

def preprocess_pipeline(df, target_column, exclude_columns=None):
    df = df.copy()
    
    if exclude_columns:
        exclude_columns = [col for col in exclude_columns if col != target_column and col in df.columns]
        if exclude_columns:
            print(f"Excluding columns: {exclude_columns}")
            df = df.drop(columns=exclude_columns)
    
    df = handle_missing_values(df)
    
    feature_cols = [col for col in df.columns if col != target_column and pd.api.types.is_numeric_dtype(df[col])]
    df = handle_outliers(df, outlier_cols=feature_cols)
    
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    X, feature_encoders = encode_features(X)
    y, target_encoder = encode_target(y)
    
    scaler = StandardScaler()
    X_scaled = X.copy()
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        X_scaled[num_cols] = scaler.fit_transform(X[num_cols])
    
    return X_scaled, y, feature_encoders, target_encoder, scaler

def get_problem_type(y):
    if len(np.unique(y)) <= 20:
        return 'classification'
    else:
        return 'regression'

def get_model(algo, problem_type):
    if problem_type == 'classification':
        return {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "svm": SVC(probability=True, random_state=42),
            "xgboost": xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
            "knn": KNeighborsClassifier(),
            "naive_bayes": GaussianNB(),
            "decision_tree": DecisionTreeClassifier(random_state=42),
            "gradient_boosting": GradientBoostingClassifier(random_state=42)
        }[algo]
    else:
        return {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "svm": SVR(),
            "xgboost": xgb.XGBRegressor(random_state=42),
            "logistic_regression": LinearRegression(),
            "knn": KNeighborsRegressor(),
            "decision_tree": DecisionTreeRegressor(random_state=42),
            "gradient_boosting": GradientBoostingRegressor(random_state=42)
        }[algo]

def feature_importance(model, X):
    if hasattr(model, "feature_importances_"):
        importances = list(zip(X.columns, model.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)
        return [{"name": name, "value": float(value)} for name, value in importances]
    
    if hasattr(model, "coef_"):
        coefs = model.coef_.flatten() if hasattr(model.coef_, "flatten") else model.coef_
        importances = list(zip(X.columns, np.abs(coefs)))
        importances.sort(key=lambda x: x[1], reverse=True)
        return [{"name": name, "value": float(value)} for name, value in importances]
    
    return []

def calculate_classification_metrics(y_true, y_pred, y_pred_proba=None, target_encoder=None):
    """Calculate comprehensive classification metrics"""
    try:
        # Get class labels
        classes = np.unique(y_true)
        
        # Get class names if encoder exists
        class_names = classes
        if target_encoder:
            try:
                class_names = target_encoder.inverse_transform(classes)
            except:
                class_names = classes
        
        # Basic metrics
        accuracy = float(accuracy_score(y_true, y_pred))
        
        # Handle multiclass vs binary
        is_binary = len(classes) == 2
        avg_method = 'binary' if is_binary else 'weighted'
        
        precision = float(precision_score(y_true, y_pred, average=avg_method, zero_division=0))
        recall = float(recall_score(y_true, y_pred, average=avg_method, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average=avg_method, zero_division=0))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        confusion_data = {
            "matrix": cm.tolist(),
            "labels": [str(label) for label in class_names],
            "classes": len(classes)
        }
        
        # AUC-ROC (for binary classification with probabilities)
        auc_roc = None
        if is_binary and y_pred_proba is not None:
            try:
                if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                    auc_roc = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
                else:
                    auc_roc = float(roc_auc_score(y_true, y_pred_proba))
            except:
                auc_roc = None
        
        # Classification Report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Per-class metrics
        per_class_metrics = []
        for i, class_label in enumerate(class_names):
            if str(i) in report:
                per_class_metrics.append({
                    "class": str(class_label),
                    "precision": float(report[str(i)].get('precision', 0)),
                    "recall": float(report[str(i)].get('recall', 0)),
                    "f1_score": float(report[str(i)].get('f1-score', 0)),
                    "support": int(report[str(i)].get('support', 0))
                })
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc,
            "confusion_matrix": confusion_data,
            "per_class_metrics": per_class_metrics,
            "classification_report": report
        }
        
    except Exception as e:
        print(f"Error calculating classification metrics: {e}")
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "auc_roc": None,
            "confusion_matrix": {"matrix": [], "labels": [], "classes": 0},
            "per_class_metrics": [],
            "classification_report": {}
        }

def calculate_regression_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics"""
    try:
        # Basic metrics
        r2 = float(r2_score(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        
        # MAPE (Mean Absolute Percentage Error)
        try:
            mape = float(mean_absolute_percentage_error(y_true, y_pred))
        except:
            # Calculate manually if sklearn version doesn't have MAPE
            mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100)
        
        # Additional metrics
        residuals = y_true - y_pred
        mean_residual = float(np.mean(residuals))
        std_residual = float(np.std(residuals))
        
        # Explained variance
        explained_variance = float(1 - np.var(residuals) / np.var(y_true))
        
        # Mean and range of actual values for context
        y_mean = float(np.mean(y_true))
        y_std = float(np.std(y_true))
        y_min = float(np.min(y_true))
        y_max = float(np.max(y_true))
        y_range = float(y_max - y_min)
        
        # Percentage errors
        mae_percentage = float((mae / y_mean) * 100) if y_mean != 0 else float('inf')
        rmse_percentage = float((rmse / y_mean) * 100) if y_mean != 0 else float('inf')
        
        return {
            "r2_score": r2,
            "mean_squared_error": mse,
            "root_mean_squared_error": rmse,
            "mean_absolute_error": mae,
            "mean_absolute_percentage_error": mape,
            "explained_variance": explained_variance,
            "mean_residual": mean_residual,
            "std_residual": std_residual,
            "target_statistics": {
                "mean": y_mean,
                "std": y_std,
                "min": y_min,
                "max": y_max,
                "range": y_range
            },
            "error_percentages": {
                "mae_percent": mae_percentage,
                "rmse_percent": rmse_percentage
            },
            "residual_analysis": {
                "mean": mean_residual,
                "std": std_residual,
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals))
            }
        }
        
    except Exception as e:
        print(f"Error calculating regression metrics: {e}")
        return {
            "r2_score": float(r2_score(y_true, y_pred)),
            "mean_squared_error": float(mean_squared_error(y_true, y_pred)),
            "root_mean_squared_error": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mean_absolute_error": float(mean_absolute_error(y_true, y_pred)),
            "mean_absolute_percentage_error": 0.0,
            "explained_variance": 0.0,
            "mean_residual": 0.0,
            "std_residual": 0.0,
            "target_statistics": {},
            "error_percentages": {},
            "residual_analysis": {}
        }

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"})
    
    if file and allowed_file(file.filename):
        try:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            
            try:
                df = pd.read_csv(filepath)
            except Exception as e:
                return jsonify({"success": False, "error": f"Error reading CSV: {str(e)}"})
            
            if df.empty:
                return jsonify({"success": False, "error": "CSV file is empty"})
            
            if df.shape[1] < 2:
                return jsonify({"success": False, "error": "CSV must have at least 2 columns"})
            
            sample_df = df.head(5).where(pd.notnull(df.head(5)), None)
            
            data = {
                "filepath": filepath,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "column_names": list(df.columns),
                "missing_values": int(df.isnull().sum().sum()),
                "sample_data": sample_df.to_dict(orient="records"),
                "target_suggestions": suggest_targets(df)
            }
            
            return jsonify({"success": True, "data": data})
            
        except Exception as e:
            return jsonify({"success": False, "error": f"Upload failed: {str(e)}"})
    
    return jsonify({"success": False, "error": "Invalid file format. Please upload a CSV file."})

@app.route('/api/preprocess', methods=['POST'])
def preprocess():
    try:
        req = request.json
        filepath = req['filepath']
        target_column = req['target_column']
        test_size = float(req.get('test_size', 0.2))
        problem_type = req.get('problem_type', 'auto')
        exclude_columns = req.get('exclude_columns', [])
        
        if not os.path.exists(filepath):
            return jsonify({"success": False, "error": "File not found"})
        
        df = pd.read_csv(filepath)
        
        if target_column not in df.columns:
            return jsonify({"success": False, "error": f"Target column '{target_column}' not found"})
        
        if exclude_columns:
            print(f"Excluding columns: {exclude_columns}")
        
        X, y, feature_encoders, target_encoder, scaler = preprocess_pipeline(
            df, target_column, exclude_columns=exclude_columns
        )
        
        if X.shape[1] == 0:
            return jsonify({"success": False, "error": "No features left after excluding columns"})
        
        if problem_type == 'auto':
            problem_type = get_problem_type(y)
        
        stratify = None
        if problem_type == 'classification' and len(np.unique(y)) > 1:
            unique, counts = np.unique(y, return_counts=True)
            if np.min(counts) >= 2:
                stratify = y
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify, random_state=42
        )
        
        data_info = {
            "problem_type": problem_type,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "target_classes": int(len(np.unique(y))) if problem_type == 'classification' else "N/A",
            "features_count": int(X.shape[1]),
            "excluded_count": len(exclude_columns) if exclude_columns else 0
        }
        
        return jsonify({"success": True, "data_info": data_info})
        
    except Exception as e:
        return jsonify({"success": False, "error": f"Preprocessing failed: {str(e)}"})

@app.route('/api/train', methods=['POST'])
def train():
    try:
        req = request.json
        algo = req['algorithm']
        cv_folds = int(req.get('cv_folds', 5))
        filepath = req.get('filepath')
        target_column = req.get('target_column')
        test_size = float(req.get('test_size', 0.2))
        problem_type = req.get('problem_type', 'auto')
        exclude_columns = req.get('exclude_columns', [])
        
        if not filepath or not target_column:
            return jsonify({"success": False, "error": "Missing required parameters"})
        
        if not os.path.exists(filepath):
            return jsonify({"success": False, "error": "File not found"})
        
        df = pd.read_csv(filepath)
        
        X, y, feature_encoders, target_encoder, scaler = preprocess_pipeline(
            df, target_column, exclude_columns=exclude_columns
        )
        
        if X.shape[1] == 0:
            return jsonify({"success": False, "error": "No features left after excluding columns"})
        
        if problem_type == 'auto':
            problem_type = get_problem_type(y)
        
        stratify = None
        if problem_type == 'classification' and len(np.unique(y)) > 1:
            unique, counts = np.unique(y, return_counts=True)
            if np.min(counts) >= 2:
                stratify = y
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify, random_state=42
        )
        
        model = get_model(algo, problem_type)
        
        import time
        t0 = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - t0
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Get probabilities for classification
        train_pred_proba = None
        test_pred_proba = None
        if problem_type == 'classification' and hasattr(model, 'predict_proba'):
            try:
                train_pred_proba = model.predict_proba(X_train)
                test_pred_proba = model.predict_proba(X_test)
            except:
                pass
        
        # Calculate comprehensive metrics
        if problem_type == 'classification':
            train_metrics = calculate_classification_metrics(y_train, train_pred, train_pred_proba, target_encoder)
            test_metrics = calculate_classification_metrics(y_test, test_pred, test_pred_proba, target_encoder)
            primary_score = test_metrics['accuracy']
            score_name = "Accuracy"
        else:
            train_metrics = calculate_regression_metrics(y_train, train_pred)
            test_metrics = calculate_regression_metrics(y_test, test_pred)
            primary_score = test_metrics['r2_score']
            score_name = "R² Score"
        
        # Cross-validation
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=min(cv_folds, len(X_train)))
            cv_score = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))
        except Exception as e:
            print(f"CV Error: {e}")
            cv_score = float(primary_score)
            cv_std = 0.0
        
        # Feature importance
        importances = feature_importance(model, pd.DataFrame(X_train, columns=X.columns))
        
        results = {
            "algorithm": algo,
            "problem_type": problem_type,
            "primary_score": float(primary_score),
            "score_name": score_name,
            "cv_score": cv_score,
            "cv_std": cv_std,
            "training_time": float(training_time),
            "feature_importance": importances,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "features_used": int(X.shape[1]),
            "excluded_columns": exclude_columns,
            # Backward compatibility
            "train_score": float(train_metrics.get('accuracy', train_metrics.get('r2_score', 0))),
            "test_score": float(primary_score),
            "overfitting": float(train_metrics.get('accuracy', train_metrics.get('r2_score', 0)) - primary_score)
        }
        
        return jsonify({"success": True, "results": results})
        
    except Exception as e:
        return jsonify({"success": False, "error": f"Training failed: {str(e)}"})

@app.route('/api/compare', methods=['POST'])
def compare():
    try:
        req = request.json
        algorithms = req['algorithms']
        cv_folds = int(req.get('cv_folds', 5))
        filepath = req.get('filepath')
        target_column = req.get('target_column')
        test_size = float(req.get('test_size', 0.2))
        problem_type = req.get('problem_type', 'auto')
        exclude_columns = req.get('exclude_columns', [])
        
        if not filepath or not target_column or not algorithms:
            return jsonify({"success": False, "error": "Missing required parameters"})
        
        if not os.path.exists(filepath):
            return jsonify({"success": False, "error": "File not found"})
        
        df = pd.read_csv(filepath)
        
        X, y, feature_encoders, target_encoder, scaler = preprocess_pipeline(
            df, target_column, exclude_columns=exclude_columns
        )
        
        if X.shape[1] == 0:
            return jsonify({"success": False, "error": "No features left after excluding columns"})
        
        if problem_type == 'auto':
            problem_type = get_problem_type(y)
        
        stratify = None
        if problem_type == 'classification' and len(np.unique(y)) > 1:
            unique, counts = np.unique(y, return_counts=True)
            if np.min(counts) >= 2:
                stratify = y
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify, random_state=42
        )
        
        results = []
        
        for algo in algorithms:
            try:
                model = get_model(algo, problem_type)
                
                import time
                t0 = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - t0
                
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Get probabilities for classification
                test_pred_proba = None
                if problem_type == 'classification' and hasattr(model, 'predict_proba'):
                    try:
                        test_pred_proba = model.predict_proba(X_test)
                    except:
                        pass
                
                # Calculate metrics
                if problem_type == 'classification':
                    train_metrics = calculate_classification_metrics(y_train, train_pred, target_encoder=target_encoder)
                    test_metrics = calculate_classification_metrics(y_test, test_pred, test_pred_proba, target_encoder)
                    primary_score = test_metrics['accuracy']
                    train_primary = train_metrics['accuracy']
                else:
                    train_metrics = calculate_regression_metrics(y_train, train_pred)
                    test_metrics = calculate_regression_metrics(y_test, test_pred)
                    primary_score = test_metrics['r2_score']
                    train_primary = train_metrics['r2_score']
                
                # Cross-validation
                try:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=min(cv_folds, len(X_train)))
                    cv_score = float(np.mean(cv_scores))
                except:
                    cv_score = float(primary_score)
                
                results.append({
                    "algorithm": algo,
                    "primary_score": float(primary_score),
                    "train_primary": float(train_primary),
                    "cv_score": cv_score,
                    "training_time": float(training_time),
                    "overfitting": float(train_primary - primary_score),
                    "detailed_metrics": test_metrics,
                    # Backward compatibility
                    "train_score": float(train_primary),
                    "test_score": float(primary_score)
                })
                
            except Exception as e:
                print(f"Error with algorithm {algo}: {e}")
                continue
        
        if not results:
            return jsonify({"success": False, "error": "All algorithms failed to train"})
        
        # Sort by primary score (descending)
        results = sorted(results, key=lambda x: x['primary_score'], reverse=True)
        best_algorithm = results[0]['algorithm']
        
        return jsonify({
            "success": True, 
            "results": results, 
            "best_algorithm": best_algorithm,
            "problem_type": problem_type,
            "features_used": int(X.shape[1]),
            "excluded_columns": exclude_columns
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": f"Comparison failed: {str(e)}"})

@app.route('/api/reset', methods=['POST'])
def reset():
    try:
        cleaned_files = 0
        for f in os.listdir(UPLOAD_FOLDER):
            if f.endswith('.csv'):
                try:
                    file_path = os.path.join(UPLOAD_FOLDER, f)
                    os.remove(file_path)
                    cleaned_files += 1
                except Exception as e:
                    print(f"Error removing file {f}: {e}")
                    pass
        
        return jsonify({"success": True, "message": f"Cleaned {cleaned_files} files"})
        
    except Exception as e:
        return jsonify({"success": False, "error": f"Reset failed: {str(e)}"})

@app.route('/')
def index():
    return """
    <h1>ML Algorithm Showcase API with Comprehensive Metrics</h1>
    <p>API is running successfully!</p>
    <h3>Available endpoints:</h3>
    <ul>
        <li>POST /api/upload - Upload CSV file</li>
        <li>POST /api/preprocess - Preprocess data with exclude columns support</li>
        <li>POST /api/train - Train single algorithm with detailed metrics</li>
        <li>POST /api/compare - Compare multiple algorithms with metrics</li>
        <li>POST /api/reset - Clean up uploaded files</li>
    </ul>
    <h3>Supported Metrics:</h3>
    <h4>Classification:</h4>
    <ul>
        <li>Accuracy, Precision, Recall, F1-Score</li>
        <li>Confusion Matrix</li>
        <li>AUC-ROC (binary classification)</li>
        <li>Per-class metrics</li>
    </ul>
    <h4>Regression:</h4>
    <ul>
        <li>R² Score, MSE, RMSE, MAE</li>
        <li>Mean Absolute Percentage Error (MAPE)</li>
        <li>Residual Analysis</li>
        <li>Target Statistics</li>
    </ul>
    """

@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)