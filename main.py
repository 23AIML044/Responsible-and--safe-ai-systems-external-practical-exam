import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "german_credit.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("RESPONSIBLE AND SAFE AI - SET B")
print("Explainable AI for EMI Finance Decisions")
print("=" * 60)

# TASK 1: LOAD AND PREPROCESS DATASET

print("\nTASK 1: Loading Dataset")

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

print("Dataset Shape:", df.shape)
print("\nDataset Preview:")
print(df.head())

if "class" not in df.columns:
    raise ValueError("Target column 'class' not found in dataset.")

df = df.dropna(subset=["class"]).copy()

target = (
    df["class"]
    .astype(str)
    .str.strip()
    .str.lower()
)

target_mapping = {
    "1": 0,
    "2": 1,
    "good": 0,
    "bad": 1
}

y = target.map(target_mapping)

if y.isna().any():
    raise ValueError("Unknown target labels found in class column.")

X = df.drop("class", axis=1)

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

for column in numeric_columns:
    X[column] = pd.to_numeric(X[column], errors="coerce")

X = X.fillna(X.median(numeric_only=True))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

if categorical_columns:
    X_train_encoded[categorical_columns] = encoder.fit_transform(
        X_train[categorical_columns].astype(str)
    )

    X_test_encoded[categorical_columns] = encoder.transform(
        X_test[categorical_columns].astype(str)
    )

feature_names = X_train_encoded.columns.tolist()

categorical_indices = [
    feature_names.index(column)
    for column in categorical_columns
]

categorical_names = {}

for i, column in enumerate(categorical_columns):
    feature_index = feature_names.index(column)

    categorical_names[feature_index] = [
        str(value) for value in encoder.categories_[i]
    ]

print("\nTotal Records:", len(df))
print("Training Records:", len(X_train_encoded))
print("Testing Records:", len(X_test_encoded))
print("Total Features:", len(feature_names))

print("\nTarget Distribution:")
print(y.value_counts().rename(index={0: "Good", 1: "Bad"}))

print("\nTask 1 Completed Successfully!")


# TASK 2: TRAIN RANDOM FOREST MODEL

print("\nTASK 2: Training Random Forest Model")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

model.fit(X_train_encoded, y_train)

y_pred = model.predict(X_test_encoded)

y_probability = model.predict_proba(X_test_encoded)[:, 1]

accuracy = accuracy_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_probability)

print("\nModel Performance")
print("-------------------------")
print("Accuracy:", round(accuracy, 4))
print("ROC-AUC Score:", round(roc_auc, 4))

print("\nTask 2 Completed Successfully!")


# TASK 3: GLOBAL SHAP EXPLANATION

print("\nTASK 3: SHAP Global Explanation")

explainer = shap.TreeExplainer(model)

shap_values_raw = explainer.shap_values(X_test_encoded)

def get_class_shap_values(values, class_index=1):
    if isinstance(values, list):
        return np.asarray(values[class_index])

    values = np.asarray(values)

    if values.ndim == 3:
        return values[:, :, class_index]

    return values

shap_values = get_class_shap_values(shap_values_raw)

global_importance = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Importance": global_importance
})

importance_df = importance_df.sort_values(
    by="SHAP Importance",
    ascending=False
)

print("\nTop 5 Important Features:")
print(importance_df.head(5).to_string(index=False))

importance_df.to_csv(
    os.path.join(RESULTS_DIR, "shap_feature_importance.csv"),
    index=False
)

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_encoded,
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "shap_summary.png"),
    bbox_inches="tight"
)

plt.close()

print("\nSHAP Summary Plot Saved!")

print("\nTask 3 Completed Successfully!")


# TASK 4: LOCAL SHAP EXPLANATION FOR REJECTED APPLICANT

print("\nTASK 4: Individual Applicant SHAP Explanation")

rejected_indices = np.where(y_pred == 1)[0]

if len(rejected_indices) == 0:
    applicant_index = int(np.argmax(y_probability))
    print("No applicant was predicted as Bad.")
    print("Showing applicant with the highest Bad probability instead.")
else:
    applicant_index = int(rejected_indices[0])

applicant = X_test_encoded.iloc[applicant_index]

applicant_probability = y_probability[applicant_index]

print("\nSelected Applicant:")
print("Test Index:", applicant_index)
print("Predicted Class:", "Bad / Rejected")
print("Probability of Bad:", round(applicant_probability, 4))

local_shap_values = shap_values[applicant_index]

if isinstance(explainer.expected_value, (list, np.ndarray)):
    base_value = np.asarray(explainer.expected_value).flatten()[1]
else:
    base_value = float(explainer.expected_value)

local_explanation = shap.Explanation(
    values=local_shap_values,
    base_values=base_value,
    data=applicant.values,
    feature_names=feature_names
)

plt.figure()

shap.plots.waterfall(
    local_explanation,
    max_display=10,
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "shap_waterfall.png"),
    bbox_inches="tight"
)

plt.close()

local_importance = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Value": local_shap_values
})

local_importance["Absolute Value"] = abs(
    local_importance["SHAP Value"]
)

local_importance = local_importance.sort_values(
    by="Absolute Value",
    ascending=False
)

print("\nTop 3 SHAP Reasons:")

for _, row in local_importance.head(3).iterrows():
    direction = (
        "Increases rejection risk"
        if row["SHAP Value"] > 0
        else "Decreases rejection risk"
    )

    print(
        row["Feature"],
        "| SHAP:",
        round(row["SHAP Value"], 4),
        "|",
        direction
    )

print("\nSHAP Waterfall Plot Saved!")

print("\nTask 4 Completed Successfully!")


# TASK 5: LIME EXPLANATION

print("\nTASK 5: LIME Local Explanation")

lime_explainer = LimeTabularExplainer(
    training_data=X_train_encoded.values,
    feature_names=feature_names,
    class_names=["Good", "Bad"],
    categorical_features=categorical_indices,
    categorical_names=categorical_names,
    mode="classification",
    random_state=42
)

lime_explanation = lime_explainer.explain_instance(
    data_row=applicant.values,
    predict_fn=model.predict_proba,
    num_features=5,
    labels=(1,)
)

print("\nTop 5 LIME Features:")

lime_features = lime_explanation.as_map()[1]

lime_features = sorted(
    lime_features,
    key=lambda x: abs(x[1]),
    reverse=True
)

lime_top_features = []

for feature_index, weight in lime_features[:5]:
    feature_name = feature_names[feature_index]

    lime_top_features.append(feature_name)

    print(
        feature_name,
        "| LIME Weight:",
        round(weight, 4)
    )

fig = lime_explanation.as_pyplot_figure(label=1)

fig.tight_layout()

fig.savefig(
    os.path.join(RESULTS_DIR, "lime_explanation.png"),
    bbox_inches="tight"
)

plt.close(fig)

shap_top_features = local_importance.head(3)["Feature"].tolist()

lime_top_three = lime_top_features[:3]

common_features = set(shap_top_features).intersection(
    set(lime_top_three)
)

print("\nSHAP Top 3 Features:", shap_top_features)
print("LIME Top 3 Features:", lime_top_three)

print(
    "Top 3 Feature Agreement:",
    len(common_features),
    "/ 3"
)

print("Common Features:", list(common_features))

print("\nLIME Explanation Plot Saved!")

print("\nTask 5 Completed Successfully!")


# TASK 6: FAIRNESS CHECK AND REJECTION MESSAGE

print("\nTASK 6: Fairness and Responsible AI Check")

sensitive_attributes = [
    "age",
    "personal_status",
    "foreign_worker"
]

shap_features = set(shap_top_features)
lime_features_set = set(lime_top_three)

all_explanation_features = shap_features.union(
    lime_features_set
)

sensitive_found = [
    feature
    for feature in sensitive_attributes
    if feature in all_explanation_features
]

print("\nSensitive Attributes Found in Top Explanations:")

if sensitive_found:
    print(sensitive_found)
    print("FAIRNESS STATUS: REVIEW REQUIRED")
else:
    print("No listed sensitive attributes found in the top 3 explanations.")
    print("FAIRNESS STATUS: No flag from this limited check.")

print("\nPlain Language Rejection Message:")

legitimate_features = [
    feature
    for feature in local_importance["Feature"]
    if feature not in sensitive_attributes
    and local_importance.loc[
        local_importance["Feature"] == feature,
        "SHAP Value"
    ].iloc[0] > 0
]

if legitimate_features:
    message_features = legitimate_features[:2]

    feature_messages = []

    for feature in message_features:
        original_value = X_test.iloc[
            applicant_index
        ][feature]

        if feature == "checking_status":
            description = "your current checking account status"

        elif feature == "duration":
            description = "the requested loan duration"

        elif feature == "credit_amount":
            description = "the requested credit amount"

        elif feature == "savings_status":
            description = "your savings status"

        elif feature == "purpose":
            description = "the stated loan purpose"

        elif feature == "employment":
            description = "your employment information"

        else:
            description = feature.replace("_", " ")

        feature_messages.append(
            description + " (recorded value: " + str(original_value) + ")"
        )

    message = (
        "Your EMI request was not approved. "
        "The model identified the following factors as contributing "
        "to the decision: "
        + " and ".join(feature_messages)
        + ". Please contact the finance provider if you would like "
        "a detailed review of your application."
    )

else:
    message = (
        "Your EMI request was not approved based on the model's "
        "overall assessment. Please contact the finance provider "
        "for a detailed review of your application."
    )

print(message)

print("\nImportant Responsible AI Note:")
print(
    "Sensitive attributes should not be used unfairly in lending decisions. "
    "A human review and a proper fairness audit are required."
)

print("\nTask 6 Completed Successfully")



