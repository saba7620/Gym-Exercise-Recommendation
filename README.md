                                                 Gym Exercise Recommendation System
Overview

This project implements a machine learning-based recommendation system that suggests gym exercises based on the target muscle group and available equipment. The system takes in user input for the desired muscle group and the type of equipment accessible, and recommends the most suitable exercises accordingly.
The goal is to help gym-goers or trainers find exercises tailored to their needs, maximizing workout efficiency and safety.

Dataset

The dataset used for this project is gym_exercise_dataset.csv, which includes various features related to gym exercises, such as:
Exercise Name: The name of the exercise.
Equipment: The equipment required for the exercise (e.g., machine, body weight).
Target Muscles: The primary muscle groups targeted by the exercise.
Synergist Muscles: Supporting muscles used during the exercise.
Main Muscle: The main muscle targeted by the exercise.

Project Structure

The project consists of the following key files:
gym_exercise_recommendation.py: Python script containing the complete code for data loading, preprocessing, feature extraction, machine learning model training, and recommendation logic.
gym_exercise_dataset.csv: The dataset used for making exercise recommendations.
README.md: Documentation file for the project.

Requirements

To run this project, you need to have Python installed along with the following packages. You can install them using the following command:
 
 pip install numpy pandas scikit-learn
