# Gym Exercise Recommendation System

## Overview

This project implements a recommendation system for gym exercises based on a dataset of machine-based exercises. The goal is to help users select exercises tailored to their desired target muscle groups and available equipment using a K-Nearest Neighbors (KNN) model.

## Dataset

The dataset includes detailed information about various gym exercises, including:
- **Exercise Name**: Name of the exercise.
- **Target Muscles**: Muscle groups targeted by the exercise.
- **Main Muscle**: Primary muscle engaged in the exercise.
- **Synergist Muscles**: Secondary muscles involved in the exercise.
- **Equipment**: The type of equipment used for the exercise (e.g., Stretch, Cable Machine, etc.).

### Dataset File:
- `gym_exercise_dataset.csv`: Contains all exercise details.

## Project Structure

- `gym_exercise_recommendation.py`: Python script containing the code for data preprocessing, feature engineering, model building, and the recommendation system.
- `gym_exercise_dataset.csv`: The dataset used for the recommendation system.
- `README.md`: Project documentation (this file).

## Features

1. **Data Preprocessing**: The dataset is cleaned, and missing values are handled. Muscles and equipment are processed into useful formats for modeling.
2. **Feature Engineering**: Target muscles and main muscles are combined to create a `Muscles` feature, and equipment is numerically encoded.
3. **Model Building**: A K-Nearest Neighbors (KNN) model is trained to recommend exercises based on the input of muscles and available equipment.
4. **User Interaction**: The script allows users to input their desired muscle group and available equipment, and it returns recommended exercises.

## Requirements

To run this project, you'll need the following libraries installed:

```bash
pip install pandas scikit-learn
