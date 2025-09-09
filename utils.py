import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import openrouteservice
import requests
import folium
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union

# 1. Clean and load data
def load_and_clean_crime_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df.dropna(subset=['incident_datetime', 'latitude', 'longitude'])
    df['incident_datetime'] = pd.to_datetime(df['incident_datetime'])
    # ...any other cleaning
    return df

# 2. Risk category assignment
high_risk_categories = {...}
high_risk_subcategories = {...}

def assign_risk(row):
    if (row['incident_subcategory'] in high_risk_subcategories) or (row['incident_category'] in high_risk_categories):
        return 1  # High-risk
    return 0

def add_risk_labels(df):
    df['risk_level'] = df.apply(assign_risk, axis=1)
    return df

# 3. Feature engineering
def add_time_features(df):
    df['incident_time'] = df['incident_time'].astype(str)
    df['incident_time_dt'] = pd.to_datetime(df['incident_time'], format='%H:%M', errors='coerce')
    df['incident_hour'] = df['incident_time_dt'].dt.hour
    df['incident_minute'] = df['incident_time_dt'].dt.minute
    df = df.dropna(subset=['incident_hour', 'incident_minute'])
    return df

def encode_day_of_week(df):
    le = LabelEncoder()
    df['day_of_week_encoded'] = le.fit_transform(df['incident_day_of_week'])
    return df, le

def one_hot_encode_day_of_week(df):
    ohe = OneHotEncoder(sparse_output=False, categories='auto')
    day_encoded = ohe.fit_transform(df[['day_of_week_encoded']])
    day_labels = ohe.get_feature_names_out(['day_of_week_encoded'])
    day_df = pd.DataFrame(day_encoded, columns=day_labels, index=df.index)
    return day_df, ohe, day_labels

def compute_severity(subcat):
    subcat = str(subcat).lower()
    # ...same logic as before

def add_severity_weights(df):
    df['severity_weight'] = df['incident_subcategory'].apply(compute_severity)
    return df

def add_recency_weights(df):
    most_recent = df['incident_datetime'].max()
    df['days_ago'] = (most_recent - df['incident_datetime']).dt.days
    df['recency_weight'] = 1 / (1 + (df['days_ago'] / 180))
    return df

def combine_weights(df):
    df['weight'] = df['recency_weight'] * df['severity_weight']
    df['weight'] = np.minimum(df['weight'], 5)
    return df

# One big wrapper for all preprocessing
def preprocess_crime_data(csv_path):
    df = load_and_clean_crime_data(csv_path)
    df = add_risk_labels(df)
    df = add_time_features(df)
    df, le = encode_day_of_week(df)
    day_df, ohe, day_labels = one_hot_encode_day_of_week(df)
    df = add_severity_weights(df)
    df = add_recency_weights(df)
    df = combine_weights(df)
    return df, day_df, le, ohe, day_labels


# 1. ORS client setup (call this in your app)
def create_ors_client(api_key):
    return openrouteservice.Client(key=api_key)

# 2. Geocoding function
def geocode_address(address, api_key):
    url = "https://api.openrouteservice.org/geocode/search"
    params = {
        "api_key": api_key,
        "text": address,
        "boundary.country": "US",
        "size": 1
    }
    response = requests.get(url, params=params)
    data = response.json()
    try:
        coords = data['features'][0]['geometry']['coordinates']  # [lon, lat]
        return tuple(coords)
    except (IndexError, KeyError):
        print(f"❌ Could not geocode: {address}")
        return None

# 3. Fetch route geometry from ORS
def get_route_coords(start, end, ors_client):
    coords = [start, end]  # (lon, lat) pairs
    try:
        route = ors_client.directions(coords, profile='foot-walking', format='geojson')
        return route['features'][0]['geometry']['coordinates']  # list of [lon, lat]
    except Exception as e:
        print(f"❌ ORS error: {e}")
        return None

# 4. Plot route on Folium map
def plot_route_on_map(coords, start_coords, end_coords, risk_score, risk_per_point, rerouted=False):
    # Flip coords for folium: (lat, lon)
    latlon_coords = [(lat, lon) for lon, lat in coords]

    # Create map centered on start
    m = folium.Map(location=[start_coords[1], start_coords[0]], zoom_start=14)

    # Polyline for route
    color = "red" if rerouted else "blue"
    folium.PolyLine(latlon_coords, color=color, weight=5, opacity=0.8).add_to(m)

    # Mark start and end
    folium.Marker(latlon_coords[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(latlon_coords[-1], popup="End", icon=folium.Icon(color="orange")).add_to(m)

    # Midpoint marker with average risk
    folium.Marker(
        location=latlon_coords[len(latlon_coords) // 2],
        popup=f"Avg Risk: {risk_score:.2f}",
        icon=folium.Icon(color="red" if rerouted else "blue")
    ).add_to(m)

    # Per-point risk markers (subtle)
    for (lat, lon), risk in zip(latlon_coords, risk_per_point):
        folium.CircleMarker(
            location=(lat, lon),
            radius=4,
            fill=True,
            fill_opacity=0.6,
            color="crimson" if risk > 0.5 else "gray",
            tooltip=f"Risk: {risk:.2f}"
        ).add_to(m)

    return m

# 5. Utility: round to nearest 15 min
def round_to_nearest_15(minute):
    return int(round(minute / 15.0) * 15) % 60

# 6. Utility: day index lookup
def day_index(day_str):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return days.index(day_str)

# 7. Score a route using classifier
def score_route(coords, hour, minute, day_str, clf, ohe, day_labels):
    minute_rounded = round_to_nearest_15(minute)
    idx = day_index(day_str)
    day_encoded_array = ohe.transform(pd.DataFrame({'day_of_week_encoded': [idx]}))
    day_vector = day_encoded_array.flatten().tolist()

    risks = []
    for lon, lat in coords:
        features = [hour, minute_rounded, lat, lon] + day_vector
        columns = ['incident_hour', 'incident_minute', 'latitude', 'longitude'] + list(day_labels)
        if len(features) != len(columns):
            print(f"Feature mismatch: {len(features)} features vs {len(columns)} columns")
            continue
        row = pd.DataFrame([features], columns=columns)
        prob = clf.predict_proba(row)[0, 1]
        risks.append(prob)

    avg_risk = sum(risks) / len(risks) if risks else 0
    return avg_risk, risks

# 8. Iterative reroute to minimize risk
def iterative_reroute_min_risk(
    coords, start, end, hour, minute, day_str,
    clf, ohe, day_labels, ors_client,
    buffer_sizes=[0.001, 0.0015, 0.002], risk_threshold=0.5
):
    idx = day_index(day_str)
    day_vector = ohe.transform(pd.DataFrame({'day_of_week_encoded': [idx]})).flatten()
    day_cols = ['incident_hour', 'incident_minute', 'latitude', 'longitude'] + list(day_labels)

    # Score original route
    original_scores = []
    for lon, lat in coords:
        features = [hour, minute, lat, lon] + day_vector.tolist()
        row = pd.DataFrame([features], columns=day_cols)
        prob = clf.predict_proba(row)[0, 1]
        original_scores.append(prob)

    original_risk = sum(original_scores) / len(original_scores)
    best_risk = original_risk
    best_coords = coords
    best_scores = original_scores
    best_buffer = None

    for buffer_size in buffer_sizes:
        try:
            # Identify top 20% riskiest points
            scores = []
            for lon, lat in coords:
                features = [hour, minute, lat, lon] + day_vector.tolist()
                row = pd.DataFrame([features], columns=day_cols)
                prob = clf.predict_proba(row)[0, 1]
                scores.append(prob)
            top_idxs = np.argsort(scores)[-int(len(scores) * 0.2):]
            avoid_coords = [[coords[i][0], coords[i][1]] for i in top_idxs]

            # Buffer and merge
            polygons = [
                Polygon([
                    (lon + buffer_size, lat + buffer_size),
                    (lon - buffer_size, lat + buffer_size),
                    (lon - buffer_size, lat - buffer_size),
                    (lon + buffer_size, lat - buffer_size),
                    (lon + buffer_size, lat + buffer_size)
                ])
                for lon, lat in avoid_coords
            ]
            merged_polygon = unary_union(polygons)
            avoid_geojson = mapping(merged_polygon)

            # ORS call for rerouted path
            route = ors_client.directions(
                coordinates=[start, end],
                profile='foot-walking',
                format='geojson',
                options={"avoid_polygons": avoid_geojson}
            )
            new_coords = route['features'][0]['geometry']['coordinates']

            # Score new route
            new_scores = []
            for lon, lat in new_coords:
                features = [hour, minute, lat, lon] + day_vector.tolist()
                row = pd.DataFrame([features], columns=day_cols)
                prob = clf.predict_proba(row)[0, 1]
                new_scores.append(prob)

            avg_risk = sum(new_scores) / len(new_scores)
            if avg_risk < best_risk:
                best_risk = avg_risk
                best_coords = new_coords
                best_scores = new_scores
                best_buffer = buffer_size
        except Exception as e:
            print(f"Reroute error (buffer={buffer_size}): {e}")
            continue

    return {
        "coords": best_coords,
        "avg_risk": best_risk,
        "risk_per_point": best_scores,
        "was_rerouted": best_coords != coords,
        "buffer_used": best_buffer,
        "original_risk": original_risk
    }

# 9. End-to-end route assessment (wrapper for above)
def assess_route(start, end, hour, minute, day_str, clf, ohe, day_labels, ors_client, threshold=0.5):
    coords = get_route_coords(start, end, ors_client)
    if coords is None:
        return None, None, None

    avg_risk, risk_per_point = score_route(coords, hour, minute, day_str, clf, ohe, day_labels)
    if avg_risk > threshold:
        print(f"⚠️ Route risk ({avg_risk:.2f}) exceeds threshold {threshold} — rerouting...")
    else:
        print(f"✅ Route is safe with risk score: {avg_risk:.2f}")

    return coords, avg_risk, risk_per_point