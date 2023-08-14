import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from sklearn.metrics import precision_score, recall_score


def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Will the Waze User Churn? 🚗")
    st.sidebar.markdown("Will the Waze User Churn? 🚗")

    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv("./data/waze_dataset.csv")
        data = data[(data["label"] == "retained") | (data["label"] == "churned")]
        labelencoder = LabelEncoder()
        for col in data.columns:
            data[col] = labelencoder.fit_transform(data[col])
        return data

    @st.cache_data(persist=True)
    def split(df):
        y = df.label
        x = df.drop(columns=["label"])
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=0
        )
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            confusion_matrix_fig = ConfusionMatrixDisplay.from_estimator(
                model, x_test, y_test
            ).figure_
            st.pyplot(confusion_matrix_fig)

        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            roc_curve_fig = RocCurveDisplay.from_estimator(
                model, x_test, y_test
            ).figure_
            st.pyplot(roc_curve_fig)

        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            precision_recall_fig = PrecisionRecallDisplay.from_estimator(
                model, x_test, y_test
            ).figure_
            st.pyplot(precision_recall_fig)

    df = load_data()
    class_names = ["retained", "churned"]

    x_train, x_test, y_train, y_test = split(df)

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox(
        "Classifier",
        ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"),
    )

    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        # choose parameters
        C = st.sidebar.number_input(
            "C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_SVM"
        )
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
        gamma = st.sidebar.radio(
            "Gamma (Kernel Coefficient)", ("scale", "auto"), key="gamma"
        )

        metrics = st.sidebar.multiselect(
            "What metrics to plot?",
            ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
        )

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write(
                "Precision: ",
                precision_score(y_test, y_pred, labels=class_names).round(2),
            )
            st.write(
                "Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2)
            )
            plot_metrics(metrics)

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input(
            "C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR"
        )
        max_iter = st.sidebar.slider(
            "Maximum number of iterations", 100, 500, key="max_iter"
        )

        metrics = st.sidebar.multiselect(
            "What metrics to plot?",
            ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
        )

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, penalty="l2", max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write(
                "Precision: ",
                precision_score(y_test, y_pred, labels=class_names).round(2),
            )
            st.write(
                "Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2)
            )
            plot_metrics(metrics)

    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input(
            "The number of trees in the forest", 100, 5000, step=10, key="n_estimators"
        )
        max_depth = st.sidebar.number_input(
            "The maximum depth of the tree", 1, 20, step=1, key="max_depth"
        )
        bootstrap = st.sidebar.radio(
            "Bootstrap samples when building trees", (True, False), key="bootstrap"
        )
        metrics = st.sidebar.multiselect(
            "What metrics to plot?",
            ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
        )

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                bootstrap=bootstrap,
                n_jobs=-1,
            )
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write(
                "Precision: ",
                precision_score(y_test, y_pred, labels=class_names).round(2),
            )
            st.write(
                "Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2)
            )
            plot_metrics(metrics)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Waze Data Set (Classification)")
        st.write(df)
        st.markdown(
            """The dataset provided contains information about Waze users, including their interactions and activities on the platform. The dataset has the following columns:

1. **ID:** A unique identifier for each user.
2. **label:** Indicates whether the user is "churned" (no longer using the platform) or "retained" (still using the platform). Retained: 1 Churned: 0
3. **sessions:** The number of sessions the user has had on the platform.
4. **drives:** The number of drives the user has participated in.
5. **total_sessions:** The total time (in some unit) spent in sessions by the user.
6. **n_days_after_onboarding:** The number of days after the onboarding process.
7. **total_navigations_fav1:** The total number of navigations to favorite location 1.
8. **total_navigations_fav2:** The total number of navigations to favorite location 2.
9. **driven_km_drives:** The total distance driven (in some unit) during drives.
10. **duration_minutes_drives:** The total duration (in minutes) of drives.
11. **activity_days:** The number of days the user has been active on the platform.
12. **driving_days:** The number of days the user has been active in terms of driving.
13. **device:** The type of device the user is using (e.g., Android: 0, iPhone: 1). 

Each row in the dataset represents a user's profile and activities on the Waze platform. The dataset captures various metrics related to user engagement, activity frequency, and usage patterns. The "label" column is the target variable, indicating whether a user is still retained or has churned."""
        )


if __name__ == "__main__":
    main()
