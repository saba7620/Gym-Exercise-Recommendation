import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load the dataset
file_path = 'gym_exercise_dataset.csv'  # Update this path if necessary
df = pd.read_csv(file_path)

# Display initial data overview
print("Initial data overview:")
print(df.head())

# Handle missing values
df.fillna('Unknown', inplace=True)

# Convert Target_Muscles and Synergist_Muscles into lists
df['Target_Muscles'] = df['Target_Muscles'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
df['Synergist_Muscles'] = df['Synergist_Muscles'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])

# Combine Target_Muscles and Main_muscle into a 'Muscles' column
df['Muscles'] = df['Target_Muscles'].apply(lambda x: ', '.join(x)) + ', ' + df['Main_muscle']

# Encode the equipment for modeling (convert equipment type into numeric form)
df['Equipment_encoded'] = df['Equipment'].astype('category').cat.codes

# Keep only necessary columns
df_clean = df[['Exercise Name', 'Muscles', 'Equipment_encoded']]

# Display cleaned data overview
print("Cleaned data overview:")
print(df_clean.head())

# Use TF-IDF Vectorizer to convert muscle groups into numerical features
vectorizer = TfidfVectorizer()
muscle_vectors = vectorizer.fit_transform(df_clean['Muscles'])

# Combine the muscle vectors with the equipment encoding
features = pd.concat([pd.DataFrame(muscle_vectors.toarray()), df_clean['Equipment_encoded'].reset_index(drop=True)], axis=1)

# Initialize and fit the K-Nearest Neighbors model
knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
knn_model.fit(features)

# Function to recommend exercises based on input muscle and equipment
def recommend_exercises(input_muscle, equipment):
    # Vectorize the input muscle
    input_vector = vectorizer.transform([input_muscle]).toarray()
    
    # Combine muscle vector with equipment encoding
    input_data = pd.DataFrame(input_vector).assign(Equipment_encoded=equipment)
    
    # Get the nearest neighbors (recommendations)
    distances, indices = knn_model.kneighbors(input_data)
    
    # Return the recommended exercises
    return df_clean.iloc[indices[0]][['Exercise Name', 'Muscles']]

# Simple user interaction function to get input and display recommendations
def user_interaction():
    print("Welcome to the Gym Exercise Recommendation System")
    
    # Ask the user for input
    target_muscle = input("Which muscle group do you want to target? (e.g., Shoulder, Neck): ")
    equipment = int(input("Which equipment do you have access to? (Enter 0 for Stretch, 1 for Cable Machine, etc.): "))
    
    # Get recommendations
    recommendations = recommend_exercises(target_muscle, equipment)
    
    # Display recommendations
    print("Here are the top recommended exercises for you:")
    print(recommendations)

# Run the interactive recommendation system
if __name__ == "__main__":
    user_interaction()
