from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def load_dataset(file_path):
    return pd.read_csv(file_path)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        df = load_dataset(file_path)
        return render_template('dataset.html', filename=file.filename, data=df.to_html())
    else:
        return 'File format not supported! Please upload a CSV or Excel file.'


@app.route('/scale_data/<filename>', methods=['GET', 'POST'])
def scale_data(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = load_dataset(file_path)

    if request.method == 'POST':
        scaling_method = request.form['scaling_method']

        # Select only numeric columns for scaling
        numeric_df = df.select_dtypes(include=['number'])

        if scaling_method == 'normalize':
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(numeric_df)
            df[numeric_df.columns] = scaled_data  # Update original DataFrame with scaled data
        elif scaling_method == 'standardize':
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)
            df[numeric_df.columns] = scaled_data  # Update original DataFrame with scaled data

        # Save the scaled data back to the original file
        df.to_csv(file_path, index=False)

        return render_template('dataset.html', filename=filename, data=df.to_html())

    return render_template('scale_data.html', filename=filename)


@app.route('/pca/<filename>', methods=['GET', 'POST'])
def pca(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = load_dataset(file_path)

    if request.method == 'POST':
        n_components = int(request.form['n_components'])

        # Identify numeric columns
        numeric_df = df.select_dtypes(include=['number'])

        if numeric_df.shape[1] == 0:
            return 'No numeric columns available for PCA.'

        # Standardizing the numeric data
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(numeric_df)

        # Applying PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(df_scaled)

        # Update original DataFrame with PCA results
        pca_df = pd.DataFrame(data=pca_result, columns=[f'Principal Component {i + 1}' for i in range(n_components)])

        # Optional: Drop the original numeric columns and keep only PCA components
        df = df.drop(numeric_df.columns, axis=1)  # Remove original numeric columns
        df = pd.concat([df, pca_df], axis=1)  # Add PCA components to the original DataFrame

        # Save the updated DataFrame back to the original file
        df.to_csv(file_path, index=False)

        # Calculate the correlation matrix for PCA results
        correlation_matrix = pca_df.corr()

        # Create a heatmap plot for the PCA correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        pca_correlation_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pca_correlation_' + filename + '.png')
        plt.savefig(pca_correlation_image_path)
        plt.close()

        return render_template('pca_result.html', filename=filename, data=df.to_html(), pca_image='pca_correlation_' + filename + '.png')

    return render_template('pca.html', filename=filename)


@app.route('/check_null/<filename>')
def check_null(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = load_dataset(file_path)

    null_data = df.isnull().sum()
    # Return only columns with null values
    return render_template('null_check.html', filename=filename, null_data=null_data[null_data > 0])


@app.route('/remove_null/<filename>', methods=['POST'])
def remove_null(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = load_dataset(file_path)
    if request.method == 'POST':

        fill_method = request.form.get('fill_method')

        if fill_method == 'drop':
            df = df.dropna(axis=1)  # Drop columns with any null values

        elif fill_method == 'bfill':
            df = df.bfill()
        elif fill_method == 'ffill':
            df = df.ffill()
        elif fill_method == 'mean':
            df = df.fillna(df.mean())
        elif fill_method == 'median':
            df = df.fillna(df.median())
        elif fill_method == 'mode':
            df = df.fillna(df.mode().iloc[0])

    df.to_csv(file_path, index=False)
    return render_template('dataset.html', filename=filename, data=df.to_html())


@app.route('/check_duplicates/<filename>')
def check_duplicates(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = load_dataset(file_path)

    duplicate_rows = df[df.duplicated()]
    return render_template('duplicate_check.html', filename=filename, duplicates=duplicate_rows)


@app.route('/remove_duplicates/<filename>', methods=['POST'])
def remove_duplicates(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = load_dataset(file_path)

    df = df.drop_duplicates()
    df.to_csv(file_path, index=False)
    return render_template('dataset.html', filename=filename, data=df.to_html())


@app.route('/check_correlation/<filename>')
def check_correlation(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = load_dataset(file_path)

    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Create a heatmap plot for the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    correlation_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'correlation_' + filename + '.png')
    plt.savefig(correlation_image_path)
    plt.close()

    return render_template('correlation.html', filename=filename, correlation_image='correlation_' + filename + '.png')


@app.route('/uploads/correlation_<filename>.png')
def display_correlation_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'correlation_' + filename + '.png')

# Route to apply PCA



@app.route('/uploads/pca_correlation_<filename>.png')
def display_pca_correlation_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'pca_correlation_' + filename + '.png')


import pandas as pd


def suggest_important_features(df, target_column):
    # Remove non-numeric columns (categorical data) because correlation works only on numeric data
    numeric_df = df.select_dtypes(include=[float, int])

    # Ensure the target column is numeric
    if target_column not in numeric_df.columns:
        return pd.DataFrame({'Feature': [], 'Correlation': []})  # Return empty if target is non-numeric

    # Compute the correlation matrix
    correlation_matrix = numeric_df.corr()

    # Get the correlation values for the target column
    target_correlations = correlation_matrix[target_column]

    # Sort features by their correlation with the target variable
    important_features = target_correlations.abs().sort_values(ascending=False)

    # Suggest features with a correlation higher than a certain threshold
    threshold = 0.1  # You can adjust this threshold
    suggested_features = important_features[important_features > threshold].index.tolist()

    return pd.DataFrame({
        'Feature': suggested_features,
        'Correlation': important_features[suggested_features]
    })


@app.route('/feature_selection/<filename>', methods=['GET', 'POST'])
def feature_selection(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = load_dataset(file_path)

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'suggest':
            target_column = request.form.get('target_column',
                                             df.columns[-1])  # Use the last column as the target by default
            suggested_features = suggest_important_features(df, target_column)
            return render_template('suggested_features.html', features=suggested_features, filename=filename)

        elif action == 'select':
            features = df.columns.tolist()
            return render_template('select_features.html', features=features, filename=filename)

    return render_template('feature_selection.html')


@app.route('/submit_selected_features/<filename>', methods=['POST'])
def submit_selected_features(filename):
    selected_features = request.form.getlist('features')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = load_dataset(file_path)

    if selected_features:
        new_df = df[selected_features]
        new_filename = 'selected_' + filename
        new_file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        new_df.to_csv(new_file_path, index=False)
        return render_template('dataset.html', filename=new_filename, data=new_df.to_html())

    return render_template('feature_selection.html', filename=filename)


@app.route('/submit_suggested_features/<filename>', methods=['POST'])
def submit_suggested_features(filename):
    selected_features = request.form.getlist('features')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = load_dataset(file_path)

    if selected_features:
        new_df = df[selected_features]
        new_filename = 'suggested_' + filename
        new_file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        new_df.to_csv(new_file_path, index=False)
        return render_template('dataset.html', filename=new_filename, data=new_df.to_html())

    return render_template('feature_selection_result.html', filename=filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def analyze_dataset(df, target_column):
    target = df[target_column]

    # If the target column is categorical (string or integer with few unique values), it's for classification
    if target.dtype == 'object' or len(target.unique()) <= 10:  # Adjust threshold for classification
        return 'classification'
    else:
        return 'regression'

@app.route('/model_selection/<filename>', methods=['GET', 'POST'])
def model_selection(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = load_dataset(file_path)
    columns = df.columns.tolist()
    target_column = columns[-1]
    # Analyze the dataset to decide which models are applicable
    analysis_type = analyze_dataset(df,target_column)

    allow_logistic_regression = analysis_type == 'classification'
    allow_decision_tree = True  # Decision trees can be used for both classification and regression
    allow_random_forest = True  # Random Forest can be used for both
    allow_linear_regression = analysis_type == 'regression'
    allow_ridge_regression = analysis_type == 'regression'
    allow_svm = analysis_type == 'classification'
    allow_knn = analysis_type == 'classification'
    allow_naive_bayes = analysis_type == 'classification'

    return render_template('model_selection.html',
                           allow_logistic_regression=allow_logistic_regression,
                           allow_decision_tree=allow_decision_tree,
                           allow_random_forest=allow_random_forest,
                           allow_linear_regression=allow_linear_regression,
                           allow_ridge_regression=allow_ridge_regression,
                           allow_svm=allow_svm,
                           allow_knn=allow_knn,
                           allow_naive_bayes=allow_naive_bayes)

@app.route('/apply_model', methods=['POST'])
def apply_model():
    model_name = request.form.get('model')
    # Implement logic to apply the selected model to the dataset
    # You will train the model based on the previously analyzed dataset and output results
    return f"You selected the model: {model_name}. Model training will be implemented here."
if __name__ == '__main__':
    app.run(debug=True)
