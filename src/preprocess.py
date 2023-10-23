from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

def remove_no_use_features(df):
    # Drop the 'furnished' and 'elevation' columns
    df = df.drop(['furnished', 'elevation'], axis=1)
    return df

def encode_features(df):
    # Initialize label encoders
    label_encoder_dict = {
        'town': LabelEncoder(),
        'block': LabelEncoder(),
        'street_name': LabelEncoder(),
        'flat_model': LabelEncoder(),
        'subzone': LabelEncoder(),
        'planning_area': LabelEncoder(),
        'region': LabelEncoder()
    }

    # 1. Convert rent_approval_date to numerical: months since "1970-01"
    df['rent_approval_date'] = pd.to_datetime(df['rent_approval_date'])
    df['rent_approval_date'] = (df['rent_approval_date'].dt.year - 2021) * 12 + df['rent_approval_date'].dt.month - 1

    # 2. Convert categorical variables to numerical using label encoding
    for col, encoder in label_encoder_dict.items():
        df[col] = encoder.fit_transform(df[col])

    # 3. Convert flat_type to numerical
    df['flat_type'] = df['flat_type'].replace({'executive': 1}).apply(lambda x: int(''.join(filter(str.isdigit, str(x)))))

    return df

def IQR(df):
    df_mahalanobis = df.copy()
    # Define the features to be used for Mahalanobis distance calculation
    features_for_mahalanobis = df_mahalanobis.columns.tolist()

    # Calculate the Mahalanobis distance for each data point
    mean = df_mahalanobis[features_for_mahalanobis].mean()
    inv_cov_matrix = np.linalg.inv(np.cov(df_mahalanobis[features_for_mahalanobis], rowvar=False))
    df_mahalanobis['mahalanobis'] = df_mahalanobis.apply(lambda x: mahalanobis(x[features_for_mahalanobis], mean, inv_cov_matrix), axis=1)

    # Calculate IQR for Mahalanobis distance
    Q1 = df_mahalanobis['mahalanobis'].quantile(0.25)
    Q3 = df_mahalanobis['mahalanobis'].quantile(0.75)
    IQR = Q3 - Q1

    # Identify outliers
    outliers_mahalanobis = (df_mahalanobis['mahalanobis'] < (Q1 - 1.5 * IQR)) | (df_mahalanobis['mahalanobis'] > (Q3 + 1.5 * IQR))

    # Filter out the outliers
    df_mahalanobis = df_mahalanobis[~outliers_mahalanobis]
    df_mahalanobis = df_mahalanobis.drop(columns='mahalanobis')
    
    return df_mahalanobis

def IoslationForest(df):
    # Initialize Isolation Forest model
    iso_forest_multi = IsolationForest(contamination=0.01, random_state=42)

    # Fit the model and predict outliers
    outliers_iso_forest_multi = iso_forest_multi.fit_predict(df) == -1

    # Filter out the outliers
    df_iso_forest_multi = df[~outliers_iso_forest_multi]

    return df_iso_forest_multi

def remove_outliers(df):
    df_mahalanobis = IQR(df)
    df_iso_forest_multi = IoslationForest(df)
    # Use the methods that preserve more data points
    if df_mahalanobis.shape[0] > df_iso_forest_multi.shape[0]:
        df = df_mahalanobis.copy()
    else:
        df = df_iso_forest_multi.copy()
    return df

def dimensionality_reduction(df):
    # Initialize PCA models (setting n_components=1 to get a single variable out of each group)
    pca_models = {
        'location': PCA(n_components=1),
        'flat_features': PCA(n_components=1),
        'area_features': PCA(n_components=1),
    }

    # Initialize Standard Scalers
    scalers = {
        'location': StandardScaler(),
        'flat_features': StandardScaler(),
        'area_features': StandardScaler()
    }

    # Define the variable groups
    location_features = ['town', 'block', 'street_name', 'latitude', 'longitude']
    flat_features = ['flat_type', 'flat_model', 'floor_area_sqm']
    area_features = ['subzone', 'planning_area', 'region']

    # 5. Standardize the features for PCA
    df[location_features] = scalers['location'].fit_transform(df[location_features])
    df[flat_features] = scalers['flat_features'].fit_transform(df[flat_features])
    df[area_features] = scalers['area_features'].fit_transform(df[area_features])

    # 6. Apply PCA
    df['location_pca'] = pca_models['location'].fit_transform(df[location_features])
    df['flat_features_pca'] = pca_models['flat_features'].fit_transform(df[flat_features])
    df['area_features_pca'] = pca_models['area_features'].fit_transform(df[area_features])

    # 7. Drop the original variables
    df = df.drop(location_features + flat_features + area_features, axis=1)

    return df


def calculate_distance_vectorized(lat, lon, data):
    # Initialize Nearest Neighbors model
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='haversine').fit(np.deg2rad(data[['latitude', 'longitude']].values))
    
    # Find the nearest point for each data point in the main dataset
    distances, indices = nbrs.kneighbors(np.deg2rad(np.column_stack((lat, lon))))
    
    # Convert from distance in radians to kilometers
    distances = distances * 6371
    
    return distances.flatten(), indices.flatten()


def count_within_radius_vectorized(lat, lon, data, radius):
    # Initialize Nearest Neighbors model for radius search
    nbrs_radius = NearestNeighbors(radius=radius/6371, algorithm='ball_tree', metric='haversine').fit(np.deg2rad(data[['latitude', 'longitude']].values))
    
    # Count the number of points within the radius for each data point in the main dataset
    results = nbrs_radius.radius_neighbors(np.deg2rad(np.column_stack((lat, lon))))
    
    counts = np.array([len(result) for result in results[1]])
    
    return counts

def AddAuxiliaryFeatures(df, radius=1, shopping_malls_path=None, mrt_planned_path=None, mrt_existing_path=None, primary_schools_path=None):
    assert shopping_malls_path is not None, "Please provide the path to the shopping malls dataset"
    assert mrt_planned_path is not None, "Please provide the path to the planned MRT stations dataset"
    assert mrt_existing_path is not None, "Please provide the path to the existing MRT stations dataset"
    assert primary_schools_path is not None, "Please provide the path to the primary schools dataset"
    # Load the auxiliary datasets
    shopping_malls = pd.read_csv(shopping_malls_path)
    mrt_planned = pd.read_csv(mrt_planned_path)
    mrt_existing = pd.read_csv(mrt_existing_path)
    primary_schools = pd.read_csv(primary_schools_path)

    # Show the first few rows of each dataset to understand their structure
    auxiliary_datasets = {
        "Shopping Malls": shopping_malls,
        "Planned MRT Stations": mrt_planned,
        "Existing MRT Stations": mrt_existing,
        "Primary Schools": primary_schools
    }

    # Apply the vectorized functions to calculate the additional features

    for name, dataset in auxiliary_datasets.items():
        # Calculate distance to the closest point
        distances, _ = calculate_distance_vectorized(df['latitude'], df['longitude'], dataset)
        df[f'{name} Closest Dist'] = distances
    
        # Calculate the number of points within the radius
        counts = count_within_radius_vectorized(df['latitude'], df['longitude'], dataset, radius)
        df[f'{name} Count 1km'] = counts
    
    return df

def stupidNormalize(df):
    # Normalize the lease_commence_date
    df['lease_commence_date'] = df['lease_commence_date'] / 1000
    return df
    


    
               