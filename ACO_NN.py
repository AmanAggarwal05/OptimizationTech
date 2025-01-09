import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# Define ACO Parameters
class AntColonyOptimizer:
    def _init_(self, num_ants, num_iterations, decay, alpha, beta, features, fitness_function):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.features = features
        self.fitness_function = fitness_function
        self.pheromone = np.ones(len(features))  # Initialize pheromones equally
        self.cache = {}  # Cache for fitness function results

    def run(self, X, y):
        best_solution = None
        best_score = float('inf')

        for iteration in range(self.num_iterations):
            solutions = []
            scores = []

            for _ in range(self.num_ants):
                solution = self.construct_solution()
                score = self.evaluate_solution(solution, X, y)
                solutions.append(solution)
                scores.append(score)

                # Update the best solution
                if score < best_score:
                    best_solution = solution
                    best_score = score

            # Update pheromones
            self.update_pheromones(solutions, scores)

        return best_solution, best_score

    def construct_solution(self):
        probabilities = (self.pheromone ** self.alpha) / (np.sum(self.pheromone ** self.alpha) + 1e-8)
        solution = [1 if np.random.rand() < probabilities[i] else 0 for i in range(len(self.features))]
        # Ensure at least one feature is selected
        if sum(solution) == 0:
            solution[np.random.randint(len(solution))] = 1
        return solution

    def evaluate_solution(self, solution, X, y):
        solution_tuple = tuple(solution)
        if solution_tuple in self.cache:
            return self.cache[solution_tuple]
        selected_features = [self.features[i] for i in range(len(solution)) if solution[i] == 1]
        score = self.fitness_function(X[:, selected_features], y)
        self.cache[solution_tuple] = score  # Cache the result
        return score

    def update_pheromones(self, solutions, scores):
        normalized_scores = 1.0 / (np.array(scores) + 1e-8)  # Avoid division by zero
        normalized_scores /= normalized_scores.sum()  # Normalize scores

        for i in range(len(self.features)):
            for solution, score in zip(solutions, normalized_scores):
                if solution[i] == 1:
                    self.pheromone[i] += score
            # Apply pheromone decay
            self.pheromone[i] *= (1 - self.decay)


# Neural Network Model
def create_model(input_dim, learning_rate):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model


# Fitness Function: Train NN and Return Validation Error
def fitness_function(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = create_model(input_dim=X_train.shape[1], learning_rate=0.001)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    y_pred = model.predict(X_val)
    return mean_squared_error(y_val, y_pred)


# Load and Preprocess Data
def load_data():
    # Simulate loading BTCUSD dataset (replace with actual data loading)
    num_rows = 4033
    num_features = 156
    X = np.random.rand(num_rows, num_features)  # Replace with actual feature matrix
    y = np.random.rand(num_rows)  # Replace with actual target values
    return X, y


# Main Execution
if _name_ == "_main_":
    X, y = load_data()
    features = list(range(X.shape[1]))  # Feature indices

    # Define ACO
    aco = AntColonyOptimizer(
        num_ants=10,
        num_iterations=20,
        decay=0.1,
        alpha=1.0,
        beta=2.0,
        features=features,
        fitness_function=fitness_function
    )

    # Run ACO for Feature Selection
    best_solution, best_score = aco.run(X, y)
    selected_features = [features[i] for i in range(len(best_solution)) if best_solution[i] == 1]

    print("Best Selected Features:", selected_features)
    print("Best Score (MSE):", best_score)

    # Train Final Model on Selected Features
    final_model = create_model(input_dim=len(selected_features), learning_rate=0.001)
    X_train, X_test, y_train, y_test = train_test_split(X[:, selected_features], y, test_size=0.2, random_state=42)
    final_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    y_pred = final_model.predict(X_test)

    print("Final MAE:", mean_absolute_error(y_test, y_pred))