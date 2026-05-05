import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib
matplotlib.use('Agg')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif

import tensorflow as tf
import shap

from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from xgboost import XGBClassifier

st.set_page_config(layout="wide")
st.title("🧠 Fetal Health Prediction Dashboard (ANN, Deep NN, RNN)")

@st.cache_data
def load_data():
    df = pd.read_csv("fetal_health.csv")
    X = df.drop("fetal_health", axis=1)
    y = tf.keras.utils.to_categorical(df["fetal_health"] - 1)
    y_labels = df["fetal_health"]
    return X, y, y_labels, df

X, y, y_labels, df = load_data()

X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
    X, y, y_labels, test_size=0.2, stratify=y_labels
)

y_train_labels_adj = y_train_labels - 1
y_test_labels_adj = y_test_labels - 1

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

selector = SelectKBest(score_func=f_classif, k=5)
selector.fit(X_train_scaled, y_train_labels)
selected_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_indices]

X_train_selected = X_train_scaled[:, selected_indices]
X_test_selected = X_test_scaled[:, selected_indices]

X_train_rnn_selected = X_train_selected.reshape(X_train_selected.shape[0], 1, X_train_selected.shape[1])
X_test_rnn_selected = X_test_selected.reshape(X_test_selected.shape[0], 1, X_test_selected.shape[1])

def build_and_train_models(X_train_fs, X_train_rnn_fs):
    ann = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_fs.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    ann_history = ann.fit(X_train_fs, y_train, epochs=30, verbose=0, validation_split=0.2)

    deep_nn = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_fs.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    deep_nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    deep_history = deep_nn.fit(X_train_fs, y_train, epochs=30, verbose=0, validation_split=0.2)

    rnn = tf.keras.models.Sequential([
        tf.keras.layers.SimpleRNN(64, activation='tanh', input_shape=(1, X_train_fs.shape[1])),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    rnn_history = rnn.fit(X_train_rnn_fs, y_train, epochs=30, verbose=0, validation_split=0.2)

    return ann, deep_nn, rnn, ann_history, deep_history, rnn_history

ann_model, deep_model, rnn_model, ann_hist, deep_hist, rnn_hist = build_and_train_models(
    X_train_selected, X_train_rnn_selected
)

ann_full, deep_full, rnn_full, ann_hist_full, deep_hist_full, rnn_hist_full = build_and_train_models(
    X_train_scaled,
    X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
)

tabs = st.tabs(["🔍 Prediction", "📊 EDA", "📈 Evaluation", "📉 Feature Importance", "📉 Training Curves"])

# -------------------- Prediction Tab --------------------
with tabs[0]:
    st.header("Make a Prediction")

    use_top5 = st.toggle("Use Top 5 Selected Features", value=True)
    feature_set = selected_features if use_top5 else X.columns
    selected_model_set = (ann_model, deep_model, rnn_model) if use_top5 else (ann_full, deep_full, rnn_full)

    model_choice = st.radio("Select a Model", ["ANN", "Deep NN", "RNN"], key="predict")
    selected_model = {
        "ANN": selected_model_set[0],
        "Deep NN": selected_model_set[1],
        "RNN": selected_model_set[2]
    }[model_choice]

    with st.form("prediction_form"):
        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(
                col,
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean())
            )
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_array = np.array([list(input_data.values())])
        scaled_input = scaler.transform(input_array)

        if use_top5:
            scaled_input = scaled_input[:, selected_indices]

        input_for_model = scaled_input.reshape(1, 1, -1) if model_choice == "RNN" else scaled_input

        prediction = selected_model.predict(input_for_model)
        predicted_index = np.argmax(prediction)

        class_labels = {0: "Normal", 1: "Suspect", 2: "Pathological"}
        st.success(f"Predicted Class: {class_labels[predicted_index]}")

        # ✅ FIXED SHAP
        if model_choice != "RNN":
            st.subheader("SHAP Explanation")

            background_data = X_train_selected if use_top5 else X_train_scaled
            input_features = list(feature_set)

            background_sample = shap.sample(background_data, 50)

            explainer = shap.Explainer(selected_model.predict, background_sample)
            shap_values = explainer(scaled_input)

            shap_vals = shap_values.values[0][:, predicted_index]

            fig, ax = plt.subplots(figsize=(10, 4))
            shap.bar_plot(shap_vals, feature_names=input_features, show=False)

            st.pyplot(fig)

        # LIME unchanged
        st.subheader("LIME Explanation")

        lime_explainer = LimeTabularExplainer(
            training_data=X_train_selected if use_top5 else X_train_scaled,
            feature_names=list(feature_set),
            class_names=["Normal", "Suspect", "Pathological"],
            mode='classification'
        )

        lime_model = selected_model.predict if model_choice != "RNN" else \
            lambda x: selected_model.predict(x.reshape(x.shape[0], 1, x.shape[1]))

        lime_exp = lime_explainer.explain_instance(scaled_input[0], lime_model, num_features=5)
        st.components.v1.html(lime_exp.as_html(), height=600)

# -------------------- EDA Tab --------------------
with tabs[1]:
    st.header("Exploratory Data Analysis")

    st.subheader("Dataset Overview")
    st.write(df.head())

    st.subheader("Class Distribution")
    st.bar_chart(df["fetal_health"].value_counts().sort_index())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    st.pyplot(fig)

    st.subheader("Feature vs Class (Boxplot)")
    feature_selected = st.selectbox("Choose a feature for boxplot", X.columns)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x="fetal_health", y=feature_selected, data=df)
    st.pyplot(fig)

    st.subheader("Correlation with Target")
    cor_target = df.corr()['fetal_health'].drop('fetal_health')
    st.bar_chart(cor_target.sort_values(ascending=False))

    st.subheader("KDE Plots by Class (Selected Features)")
    kde_features = st.multiselect("Select features for KDE plot", X.columns.tolist(), default=list(selected_features))
    df_scaled = df.copy()
    eda_scaler = StandardScaler()
    df_scaled[kde_features] = eda_scaler.fit_transform(df_scaled[kde_features])
    df_scaled['fetal_health_label'] = df['fetal_health'].map({1: 'Normal', 2: 'Suspect', 3: 'Pathological'})

    for col in kde_features:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.kdeplot(data=df_scaled, x=col, hue='fetal_health_label', fill=True)
        plt.title(f"KDE Plot of {col} by Fetal Health Class")
        st.pyplot(fig)

# -------------------- Evaluation Tab --------------------
with tabs[2]:
    st.header("Model Evaluation")
    model_choice_eval = st.radio("Select model for evaluation", ["ANN", "Deep NN", "RNN"], key="eval")
    use_top5_eval = st.toggle("Use Top 5 Features for Evaluation", value=True)

    eval_model = {
        "ANN": ann_model if use_top5_eval else ann_full,
        "Deep NN": deep_model if use_top5_eval else deep_full,
        "RNN": rnn_model if use_top5_eval else rnn_full
    }[model_choice_eval]

    X_eval = X_test_selected if use_top5_eval else X_test_scaled
    X_eval = X_eval.reshape(X_eval.shape[0], 1, X_eval.shape[1]) if model_choice_eval == "RNN" else X_eval

    y_pred = eval_model.predict(X_eval)
    y_pred_class = np.argmax(y_pred, axis=1)
    y_true_class = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true_class, y_pred_class)
    st.write(f"Model Accuracy: {acc:.4f}")

    st.subheader("Classification Report")
    report = classification_report(y_true_class, y_pred_class, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true_class, y_pred_class)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3], yticklabels=[1, 2, 3])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    st.subheader("ROC and AUC Curve")
    y_test_bin = label_binarize(y_true_class, classes=[0, 1, 2])
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, ax = plt.subplots()
    for i in range(3):
        plt.plot(fpr[i], tpr[i], label=f'Class {i + 1} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    st.pyplot(fig)

# ------------------------- FEATURE IMPORTANCE ----------------------------------
with tabs[3]:
    st.header("Feature Importance (Random Forest, XGBoost, SHAP, PCA, DBSCAN)")

    use_top5_importance = st.toggle("Use Top 5 Features for Importance", value=True)
    X_train_importance = X_train_selected if use_top5_importance else X_train_scaled
    feature_names_importance = list(selected_features) if use_top5_importance else list(X.columns)

    shap_plot_type = st.radio("SHAP Plot Type", ["dot", "bar"], horizontal=True)

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_importance, y_train_labels)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances[sorted_idx], y=np.array(feature_names_importance)[sorted_idx])
    plt.title("Random Forest Feature Importances")
    st.pyplot(fig)

    # SHAP for Random Forest
    st.subheader("SHAP - Random Forest")
    explainer_rf = shap.TreeExplainer(rf)
    shap_values_rf = explainer_rf.shap_values(X_train_importance)

    plt.figure(figsize=(6, 4))
    shap.summary_plot(
        shap_values_rf,
        pd.DataFrame(X_train_importance, columns=feature_names_importance),
        plot_type=shap_plot_type,
        show=False
    )
    ax = plt.gca()
    labels = ax.get_yticklabels()
    for label in labels:
        label.set_fontsize(5)
        label.set_horizontalalignment('left')
        label.set_x(-0.02)
    plt.subplots_adjust(left=0.3)
    st.pyplot(plt.gcf())

    # XGBoost
    st.subheader("XGBoost Feature Importances")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb.fit(X_train_importance, y_train_labels_adj)
    xgb_importances = xgb.feature_importances_

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=xgb_importances, y=feature_names_importance)
    plt.title("XGBoost Feature Importances")
    st.pyplot(fig)

    # SHAP for XGBoost
    st.subheader("SHAP - XGBoost")
    explainer_xgb = shap.TreeExplainer(xgb)
    shap_values_xgb = explainer_xgb.shap_values(X_train_importance)

    plt.figure(figsize=(6, 4))
    shap.summary_plot(
        shap_values_xgb,
        pd.DataFrame(X_train_importance, columns=feature_names_importance),
        plot_type=shap_plot_type,
        show=False
    )
    ax = plt.gca()
    labels = ax.get_yticklabels()
    for label in labels:
        label.set_fontsize(5)
        label.set_horizontalalignment('left')
        label.set_x(-0.02)
    plt.subplots_adjust(left=0.3)
    st.pyplot(plt.gcf())

    # PCA
    st.subheader("PCA: Principal Component Analysis")
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_train_importance)
    df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    df_pca['fetal_health'] = y_train_labels_adj.values

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='fetal_health', palette='deep')
    plt.title("PCA Scatter Plot of Fetal Health Classes")
    st.pyplot(fig)

    # DBSCAN
    st.subheader("DBSCAN Clustering on PCA")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    db_labels = dbscan.fit_predict(pca_data)
    df_pca['DBSCAN_Cluster'] = db_labels

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='DBSCAN_Cluster', palette='tab10', legend='full')
    plt.title("DBSCAN Clustering in PCA Space")
    st.pyplot(fig)

    st.write("\n**Cluster Label -1 indicates noise points (unclustered)**")

    # Outliers Table
    st.subheader("DBSCAN Outliers Table")
    original_data = X_train.reset_index(drop=True).copy()
    original_data["fetal_health"] = y_train_labels.reset_index(drop=True)
    original_data["PC1"] = df_pca["PC1"].values
    original_data["PC2"] = df_pca["PC2"].values
    original_data["DBSCAN_Cluster"] = df_pca["DBSCAN_Cluster"].values

    outliers_df = original_data[original_data["DBSCAN_Cluster"] == -1]
    st.write(f"**Outliers detected by DBSCAN: {outliers_df.shape[0]} samples**")
    st.dataframe(outliers_df.reset_index(drop=True))

    csv = outliers_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Outliers as CSV", data=csv, file_name="dbscan_outliers.csv", mime="text/csv")

# -------------------- Training Curves Tab --------------------
with tabs[4]:
    st.header("Model Training Curves")

    curve_model_choice = st.radio("Select Model", ["ANN", "Deep NN", "RNN"], key="curve")
    use_top5_curve = st.toggle("Use Top 5 Selected Features", value=True, key="toggle_curve")

    histories = {
        "ANN": ann_hist if use_top5_curve else ann_hist_full,
        "Deep NN": deep_hist if use_top5_curve else deep_hist_full,
        "RNN": rnn_hist if use_top5_curve else rnn_hist_full
    }

    history = histories[curve_model_choice]
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    st.subheader("Accuracy Over Epochs")
    fig_acc, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(epochs, acc, label='Train Accuracy')
    ax.plot(epochs, val_acc, label='Validation Accuracy')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig_acc)

    st.subheader("Loss Over Epochs")
    fig_loss, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(epochs, loss, label='Train Loss')
    ax.plot(epochs, val_loss, label='Validation Loss')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    st.pyplot(fig_loss)



