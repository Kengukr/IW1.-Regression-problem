import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

DATA_PATH = os.path.join('data', 'auto-mpg.data')

def load_data():
    """Load and perform basic data processing"""
    column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 
                    'acceleration', 'model_year', 'origin', 'car_name']
    
    df = pd.read_csv(DATA_PATH, delim_whitespace=True, names=column_names, 
                   quotechar='"', skipinitialspace=True)
    
    # Check for missing or incorrect values
    if df['horsepower'].dtype == object:
        df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

    for column in df.select_dtypes(include=[np.number]).columns:
        if df[column].isnull().sum() > 0:
            df[column].fillna(df[column].median(), inplace=True)
    
    return df


    #Custom linear regressor implementation
class CustomLinearRegressor:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    #Prepare features for the model
def prepare_features(df):
    X = df.drop(['mpg', 'car_name'], axis=1)
    y = df['mpg']
    
    # Standardize features
    scaler = StandardScaler()
    numeric_features = X.select_dtypes(include=[np.number]).columns
    X_scaled = X.copy()
    X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])
    selector = SelectKBest(f_regression, k=5)
    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = X.columns[selector.get_support()]
    print("Selected features:", selected_features)
    
    return X_scaled, y, selected_features

#Train and evaluate models
def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    custom_reg = CustomLinearRegressor(learning_rate=0.01, iterations=1000)
    custom_reg.fit(X_train.values, y_train.values)
    y_pred_custom = custom_reg.predict(X_test.values)
    
    # scikit models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'model': model, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
        print(f"{name}:")
        print(f"  MSE: {mse:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R2: {r2:.2f}")
    
    # Add custom regressor results
    mse_custom = mean_squared_error(y_test, y_pred_custom)
    rmse_custom = np.sqrt(mse_custom)
    mae_custom = mean_absolute_error(y_test, y_pred_custom)
    r2_custom = r2_score(y_test, y_pred_custom)
    
    results['Custom Regressor'] = {'model': custom_reg, 'MSE': mse_custom, 'RMSE': rmse_custom, 
                                  'MAE': mae_custom, 'R2': r2_custom}
    print(f"Custom Regressor:")
    print(f"  MSE: {mse_custom:.2f}")
    print(f"  RMSE: {rmse_custom:.2f}")
    print(f"  MAE: {mae_custom:.2f}")
    print(f"  R2: {r2_custom:.2f}")
    
    return X_train, X_test, y_train, y_test, results

def hyperparameter_tuning(X_train, y_train, best_model_name, results):
    #Hyperparameter tuning for the best model
    best_model_type = type(results[best_model_name]['model'])
    
    if best_model_type == RandomForestRegressor:
        print(f"Performing Grid Search for {best_model_name}...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestRegressor(random_state=42)
    
    elif best_model_type == GradientBoostingRegressor:
        print(f"Performing Grid Search for {best_model_name}...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
        model = GradientBoostingRegressor(random_state=42)
    
    elif best_model_type in [Ridge, Lasso]:
        print(f"Performing Grid Search for {best_model_name}...")
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        }
        model = best_model_type()
    
    else:
        print(f"Unable to perform Grid Search for {best_model_name}")
        return None
    
    grid_search = GridSearchCV(model, param_grid, cv=5, 
                              scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

def feature_importance(X, best_model):
    #Determine feature importance
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.Series(best_model.feature_importances_, index=X.columns)
        print("Feature importance:")
        for feature, importance in feature_importance.sort_values(ascending=False).items():
            print(f"  {feature}: {importance:.4f}")
        return feature_importance
    return None

def main():
    df = load_data()
    print(f"Loaded {len(df)} records")
    
    print("\nPreparing features...")
    X, y, selected_features = prepare_features(df)
    
    print("\nTraining and evaluating models...")
    X_train, X_test, y_train, y_test, results = train_and_evaluate_models(X, y)
    
    # Determine the best model by RMSE
    best_model_name = min(results, key=lambda x: results[x]['RMSE'])
    print(f"\nBest model: {best_model_name} (RMSE: {results[best_model_name]['RMSE']:.2f})")
    
    print("\nPerforming hyperparameter tuning for the best model...")
    best_model_tuned = hyperparameter_tuning(X_train, y_train, best_model_name, results)
    
    if best_model_tuned is not None:
        y_pred_tuned = best_model_tuned.predict(X_test)
        rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
        print(f"RMSE after hyperparameter tuning: {rmse_tuned:.2f}")
        
        print("\nAnalyzing feature importance...")
        feature_importance(X, best_model_tuned)
    
    print("\nAnalysis complete! yehuuu!")

if __name__ == "__main__":
    main()