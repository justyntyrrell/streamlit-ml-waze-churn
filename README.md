# Waze Churn Prediction Web App

## Introduction

Welcome to the Waze Churn Prediction Web App! This web application is built using Streamlit and machine learning algorithms to predict whether a Waze user will churn or not based on various engagement and activity metrics.

Try out the app: [Waze Churn Prediction Web App](https://waze-churn.streamlit.app/)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [File Structure](#file-structure)
- [License](#license)

## Overview

The goal of this project is to provide an interactive way to predict whether a Waze user will churn, helping to understand user behavior and enhance retention strategies. The app presents users with different classifier options, allowing them to explore how various machine learning models perform in predicting user churn.

## Features

- **Classifier Options:** Choose from different classifiers: Support Vector Machine (SVM), Logistic Regression, and Random Forest.
- **Hyperparameter Tuning:** Adjust classifier hyperparameters using sliders and radio buttons.
- **Evaluation Metrics:** Visualize key metrics such as accuracy, precision, recall, ROC curve, and precision-recall curve.
- **Interactive Interface:** User-friendly interface built with Streamlit for easy interaction.

## Usage

1. Open the [Waze Churn Prediction Web App](https://waze-churn.streamlit.app/).
2. Explore the sidebar to choose your preferred classifier and set hyperparameters.
3. Click the "Classify" button to see the classifier's predictions and evaluation metrics.
4. Visualize key metrics and compare results across different classifiers.

## How It Works

The app utilizes machine learning algorithms to predict user churn based on a dataset containing engagement and activity metrics. The user can select a classifier, adjust its hyperparameters, and see how well it predicts user churn. The app displays evaluation metrics like accuracy, precision, recall, and visualizations such as ROC curves and precision-recall curves.

## File Structure

- **app.py:** The main Streamlit app script containing the code for creating the interactive web interface and integrating machine learning models.
- **data/waze_dataset.csv:** The dataset used for training and testing the models. It contains user engagement and activity metrics.

Explore the [Waze Churn Prediction Web App](https://waze-churn.streamlit.app/) to see the app in action and interact with its features.
