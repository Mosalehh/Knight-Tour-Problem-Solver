import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
import time  # Import time library for measuring execution time

# -------------- BACKTRACKING SOLUTION --------------

def is_valid(x, y, board, n): #check if within board borders
    return 0 <= x < n and 0 <= y < n and board[x][y] == -1

def knight_tour_backtracking(n):
    board = []
    for i in range(n):
        row = [-1] * n  # Create a row with n elements, all set to -1
        board.append(row)  # Append the row to the board

    move_x = [-2, -1, 1, 2, 2, 1, -1, -2]
    move_y = [1, 2, 2, 1, -1, -2, -2, -1]

    board[0][0] = 0  #Start at (0,0) coordinates

    def findTour(x, y, movei):
        if movei == n * n: #Base case when current move is n x n
            return True

        for i in range(8):
            next_x = x + move_x[i]
            next_y = y + move_y[i]
            if is_valid(next_x, next_y, board, n):
                board[next_x][next_y] = movei
                if findTour(next_x, next_y, movei + 1):
                    return True
                board[next_x][next_y] = -1  # Backtrack
        return False

    if findTour(0, 0, 1):
        return board
    else:
        return None


# -------------- GENETIC ALGORITHM SOLUTION --------------

class KnightTourGA:
    def __init__(self, n, population_size=800, generations=5000, mutation_rate=0.45):
        self.n = n
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.move_x = [-2, -1, 1, 2, 2, 1, -1, -2]
        self.move_y = [1, 2, 2, 1, -1, -2, -2, -1]

    def valid_moves(self, x, y):
        moves = []
        for dx, dy in zip(self.move_x, self.move_y):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.n and 0 <= ny < self.n:
                moves.append((nx, ny))
        return moves

    def generate_individual(self):
        path = [(0, 0)]  # Start at (0,0)
        visited = set(path)

        while len(path) < self.n * self.n:
            moves = self.valid_moves(*path[-1])
            random.shuffle(moves)
            for move in moves:
                if move not in visited:
                    path.append(move)
                    visited.add(move)
                    break
            else:
                break  

        return path

    def fitness(self, individual):
        score = 0
        for i in range(len(individual) - 1):
            x1, y1 = individual[i]
            x2, y2 = individual[i + 1]
            if (x2 - x1, y2 - y1) in zip(self.move_x, self.move_y):
                score += 1
            else:
                break
        return score

    def crossover(self, parent1, parent2):
        size = len(parent1)
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size - 1)

        child_path = parent1[start:end]
        visited = set(child_path)

        for move in parent2:
            if move not in visited:
                child_path.append(move)
                visited.add(move)

        return child_path

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            idx1 = random.randint(0, len(individual) - 1)
            idx2 = random.randint(0, len(individual) - 1)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def evolve(self):
        population = [self.generate_individual() for _ in range(self.population_size)]

        for generation in range(self.generations):
            population.sort(key=lambda ind: self.fitness(ind), reverse=True)
            best_fit = self.fitness(population[0])

            print(f"Generation {generation+1}: Best fitness = {best_fit}/{self.n * self.n - 1}")

            if best_fit == self.n * self.n - 1:
                print("\nComplete Knight's Tour found!")
                return self.convert_path_to_board(population[0])

            next_gen = population[:10]  #Use elitism and Keep top 10 individuals

            while len(next_gen) < self.population_size:
                parent1 = random.choice(population[:50])
                parent2 = random.choice(population[:50])
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_gen.append(child)

            population = next_gen

        print("\nBest attempt after all generations:")
        return self.convert_path_to_board(population[0])

    def convert_path_to_board(self, path):
        board = [[-1 for _ in range(self.n)] for _ in range(self.n)]
        for idx, (x, y) in enumerate(path):
            board[y][x] = idx
        return board


# -------------- ANIMATION FUNCTION --------------

def animate_knight_tour(board, title, icon_path):
    n = len(board)
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.title(title, fontsize=16)

    # Create white background with black grid as board
    ax.set_facecolor('white')
    ax.set_xticks(np.arange(0, n, 1))
    ax.set_yticks(np.arange(0, n, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid(True, which='both', color='black', linewidth=1)
    plt.xlim(0, n)
    plt.ylim(0, n)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')

    # Load knight png image
    knight_img = mpimg.imread(icon_path)

    # Find the knight's move sequence
    moves = [None] * (n * n)
    for i in range(n):
        for j in range(n):
            if board[i][j] != -1:
                moves[board[i][j]] = (i, j)

    knight_artist = ax.imshow(knight_img, extent=[0, 1, 0, 1], zorder=10)
    move_texts = {}

    def update(frame):
        if frame < len(moves):
            i, j = moves[frame]

            # Move knight image with moves
            knight_artist.set_extent([j, j+1, i, i+1])

            # Add move number
            move_num = frame
            if (i, j) not in move_texts:
                txt = ax.text(j + 0.5, i + 0.5, str(move_num), color='red', ha='center', va='center', fontsize=12, weight='bold')
                move_texts[(i, j)] = txt
        else:
            # After all moves done, hide the knight to show results
            knight_artist.set_visible(False)

        return knight_artist, *move_texts.values()

    ani = animation.FuncAnimation(fig, update, frames=len(moves) + 1, interval=700, blit=True, repeat=False)

    plt.show(block=True)  # Keep window open


# --------------User input windows--------------

def get_user_inputs():
    root = tk.Tk()
    root.withdraw()

    n = simpledialog.askinteger("Board Size", "Enter board size (n x n):", minvalue=1, maxvalue=100)
    if n is None:
        exit()

    algorithm = simpledialog.askinteger("algorithm", "Choose algorithm:\n1 - Backtracking\n2 - Genetic Algorithm", minvalue=1, maxvalue=2)
    if algorithm is None:
        exit()

    return n, algorithm

# -------------- MAIN --------------

print("=== Knight's Tour Solver ===")

n, algorithm = get_user_inputs()

if n < 5:
    print("\nWarning: Knight's Tour may not have a full solution for board size < 5.\n")

#Copy path the image in G33_Phase2 folder
icon_path = r"C:\Users\Mohamed Saleh\Downloads\KTH.png"

# Start measuring execution time
start_time = time.time()

if algorithm == 1:
    print("\nSolving using Backtracking...")
    board_backtracking = knight_tour_backtracking(n)
    if board_backtracking:
        animate_knight_tour(board_backtracking, "Knight's Tour - Backtracking", icon_path)
    else:
        messagebox.showinfo("Result", "No solution found using Backtracking!")
elif algorithm == 2:
    print("\nSolving using Genetic Algorithm...")
    ktga = KnightTourGA(n)
    board_genetic = ktga.evolve()
    animate_knight_tour(board_genetic, "Knight's Tour - Genetic Algorithm", icon_path)
else:
    messagebox.showerror("Error", "Invalid choice!")

# End measuring exec time
end_time = time.time()

# Calculate and print the execution time
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.6f} seconds")
