import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data
# 1. Define the standard KDD column names
kdd_columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
    'dst_host_srv_rerror_rate', 'label'
]

# 2. Load the data WITH the column names
# Note: If your specific version of NSL-KDD has a 43rd column for 'difficulty_level', 
# add it to the end of the kdd_columns list above.
df = pd.read_csv("../data/KDDTest.csv", names=kdd_columns)

# 2. Preprocessing
encoder = LabelEncoder()
df['protocol_type'] = encoder.fit_transform(df['protocol_type'])
df['service'] = encoder.fit_transform(df['service'])
df['flag'] = encoder.fit_transform(df['flag'])

# Binarize labels: 0 for normal, 1 for attack
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

X = df.drop('label', axis=1)
y = df['label']

# 3. Data Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Define Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

trained_models = {}

# 6. Train and Evaluate Models
print("--- Model Evaluation ---")
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    y_pred = model.predict(X_test)
    
    print(f"[{name}] Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

# 7. Demo Data: Safe vs. Attack Samples
feature_names = X.columns

# SAMPLE 1: Normal HTTP Traffic (Safe)
# Short duration, standard HTTP service, normal data exchange, no errors or suspicious flags.
safe_traffic = [
    0, 1, 24, 1, 215, 45076, 0, 0, 0, 0, 
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 1, 1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 
    0.0, 255, 255, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]

# SAMPLE 2: Password Guessing / Brute Force (Attack)
# Failed logins, root shell attempts, compromised conditions, high error rates.
attack_traffic = [
    0, 1, 45, 1, 0, 0, 0, 0, 0, 0, 
    20, 0, 5, 1, 1, 5, 10, 3, 5, 0, 
    0, 0, 255, 255, 1.0, 1.0, 0.5, 0.5, 1.0, 0.0, 
    0.0, 255, 255, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5
]

demo_samples = {
    "Safe Web Browsing": safe_traffic,
    "Brute Force Attack": attack_traffic
}

print("\n=== LIVE DEMO INFERENCE ===")

for scenario_name, traffic_data in demo_samples.items():
    print(f"\nAnalyzing Scenario: {scenario_name}...")
    
    # Create DataFrame and Scale
    connection_df = pd.DataFrame([traffic_data], columns=feature_names)
    connection_scaled = scaler.transform(connection_df)
    
    attack_votes = 0
    total_models = len(trained_models)
    
    # Collect predictions from all models
    for name, model in trained_models.items():
        prediction = model.predict(connection_scaled)[0]
        
        if prediction == 1:
            attack_votes += 1
            print(f"  [-] {name}: 🔴 ATTACK")
        else:
            print(f"  [-] {name}: 🟢 NORMAL")
            
    # Final Verdict Logic (Majority Rules)
    print("  -------------------------")
    if attack_votes >= (total_models / 2):
        print(f"  ➔ FINAL VERDICT: 🚨 ATTACK DETECTED ({attack_votes}/{total_models} models flagged it)")
    else:
        print(f"  ➔ FINAL VERDICT: ✅ TRAFFIC IS SAFE ({total_models - attack_votes}/{total_models} models cleared it)")