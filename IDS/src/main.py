import pandas as pd

df = pd.read_csv("../data/KDDTest.csv")


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df['protocol_type'] = encoder.fit_transform(df['protocol_type'])
df['service'] = encoder.fit_transform(df['service'])
df['flag'] = encoder.fit_transform(df['flag'])

df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

X = df.drop('label', axis=1)
y = df['label']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

ids_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

ids_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

y_pred = ids_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


df.head(10)

import numpy as np

# SAMPLE NEW CONNECTION (1 row, same number of features as X)
# These values are realistic examples from KDD-like data
feature_names = X.columns
import pandas as pd

# new_connection_df = pd.DataFrame(
#     [[
#         0, 1, 22, 9, 491, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 255, 255, 0.00, 0.00, 0.00, 0.00,
#         1.00, 0.00, 0.00, 255, 255, 1.00, 0.00,
#         0.00, 0.00, 0.00, 0.00, 0.00, 0.00
#     ]],
#     columns=feature_names
# )

new_connection_df = pd.DataFrame(
    [[
        0,    # duration
        1,    # protocol_type
        45,   # service
        1,    # flag
        0,    # src_bytes
        0,    # dst_bytes
        0,    # land
        0,    # wrong_fragment
        0,    # urgent
        0,    # hot
        20,   # 🚨 num_failed_logins (VERY HIGH)
        0,    # logged_in
        5,    # 🚨 num_compromised
        1,    # 🚨 root_shell
        1,    # 🚨 su_attempted
        5,    # 🚨 num_root
        10,   # 🚨 num_file_creations
        3,    # 🚨 num_shells
        5,    # 🚨 num_access_files
        0,    # num_outbound_cmds
        0,    # is_host_login
        0,    # is_guest_login
        255,  # 🚨 count
        255,  # 🚨 srv_count
        1.00, # 🚨 serror_rate
        1.00, # 🚨 srv_serror_rate
        0.50, # 🚨 rerror_rate
        0.50, # 🚨 srv_rerror_rate
        1.00, # 🚨 same_srv_rate
        0.00, # diff_srv_rate
        0.00, # srv_diff_host_rate
        255,  # 🚨 dst_host_count
        255,  # 🚨 dst_host_srv_count
        1.00, # 🚨 dst_host_same_srv_rate
        0.00, # dst_host_diff_srv_rate
        0.00, # dst_host_same_src_port_rate
        1.00, # 🚨 dst_host_srv_diff_host_rate
        1.00, # 🚨 dst_host_serror_rate
        1.00, # 🚨 dst_host_srv_serror_rate
        0.50, # 🚨 dst_host_rerror_rate
        0.50  # 🚨 dst_host_srv_rerror_rate
    ]],
    columns=feature_names
)

new_connection_scaled = scaler.transform(new_connection_df)
prediction = ids_model.predict(new_connection_scaled)

if prediction[0] == 0:
    print("🟢 Prediction: NORMAL traffic")
else:
    print("🔴 Prediction: ATTACK detected")