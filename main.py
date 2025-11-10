
# pip install numpy matplotlib
# python main.py


import random
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

# Rendre les résultats reproductibles
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Dimensions du conteneur
BOX_WIDTH, BOX_HEIGHT = 10.0, 6.0

# Les rectangles qu'on veut placer, j'ai choisi des tailles un peu aléatoires
rectangles = [
    (2.2, 1.2), (1.8, 1.3), (1.6, 1.6), (1.2, 2.2),
    (2.0, 1.0), (1.2, 1.0), (1.5, 1.1), (1.3, 1.7),
    (2.0, 1.0), (1.2, 1.0), (1.5, 1.1), (1.3, 1.7),
    (1.4, 1.0), (1.0, 1.5), (1.8, 0.9), (1.1, 1.1),
    (1.4, 1.0), (1.0, 1.5), (1.8, 0.9), (1.1, 1.1)
]
num_rects = len(rectangles)

# Paramètres de l'algorithme génétique, à ajuster si besoin
population_size = 120
generations = 100
num_elites = 2  # garder les meilleurs
tournament_size = 4
crossover_prob = 0.8
mutation_prob = 0.3
mutation_strength = 0.08

# Poids des pénalités pour les mauvaises dispositions
overlap_penalty = 1000.0  # vraiment pas de chevauchements
outside_penalty = 1.0


# Chaque chromosome représente les positions de tous les rectangles
# On encode comme [x1, y1, x2, y2, ...] où chaque x,y est dans [0,1]
# et est mis à l'échelle aux dimensions réelles de la boîte
Chromosome = np.ndarray


def random_chromosome():
    # Générer une solution aléatoire
    return np.random.rand(2 * num_rects)


def decode_chromosome(chromosome):
    # Convertir le chromosome en positions réelles des rectangles
    rects = []
    for i, (w, h) in enumerate(rectangles):
        # Récupérer la position du centre depuis le chromosome
        center_x = float(chromosome[2 * i + 0]) * BOX_WIDTH
        center_y = float(chromosome[2 * i + 1]) * BOX_HEIGHT
        # Convertir en coordonnées des coins
        x1, y1 = center_x - w/2, center_y - h/2
        x2, y2 = center_x + w/2, center_y + h/2
        rects.append((x1, y1, x2, y2))
    return rects


def calculate_overlap_area(rect_a, rect_b):
    # Calcule le chevauchement entre deux rectangles
    x_overlap = max(0.0, min(rect_a[2], rect_b[2]) - max(rect_a[0], rect_b[0]))
    y_overlap = max(0.0, min(rect_a[3], rect_b[3]) - max(rect_a[1], rect_b[1]))
    return x_overlap * y_overlap


def calculate_outside_area(rect):
    # Calcule la surface du rectangle qui sort du conteneur
    x1, y1, x2, y2 = rect
    width, height = x2 - x1, y2 - y1
    total_area = width * height

    # Calculer quelle partie est dans le conteneur
    clipped_x1 = max(0.0, x1)
    clipped_y1 = max(0.0, y1)
    clipped_x2 = min(BOX_WIDTH, x2)
    clipped_y2 = min(BOX_HEIGHT, y2)

    inside_width = max(0.0, clipped_x2 - clipped_x1)
    inside_height = max(0.0, clipped_y2 - clipped_y1)
    inside_area = inside_width * inside_height

    return max(0.0, total_area - inside_area)


def fitness_score(chromosome):
    # Calculer à quel point une solution est mauvaise (plus bas = mieux)
    rect_positions = decode_chromosome(chromosome)

    # Compter tous les chevauchements
    total_overlap = 0.0
    for i in range(num_rects):
        for j in range(i + 1, num_rects):
            total_overlap += calculate_overlap_area(
                rect_positions[i], rect_positions[j])

    # Compter la surface qui sort du conteneur
    total_outside = sum(calculate_outside_area(rect)
                        for rect in rect_positions)

    return overlap_penalty * total_overlap + outside_penalty * total_outside


def tournament_selection(population):
    # Choisir un parent avec la sélection par tournoi
    contestants = random.sample(population, tournament_size)
    contestants.sort(key=fitness_score)
    return contestants[0].copy()  # retourner le meilleur


def crossover(parent1, parent2, blend_factor=0.5):
    # Mélanger deux parents pour créer une descendance
    min_vals = np.minimum(parent1, parent2)
    max_vals = np.maximum(parent1, parent2)
    range_size = max_vals - min_vals

    # Étendre un peu la plage pour l'exploration
    lower = np.clip(min_vals - blend_factor * range_size, 0.0, 1.0)
    upper = np.clip(max_vals + blend_factor * range_size, 0.0, 1.0)

    child1 = np.random.uniform(lower, upper)
    child2 = np.random.uniform(lower, upper)
    return child1, child2


def mutate_chromosome(chromosome, mutation_rate):
    # Modifier aléatoirement un chromosome
    if np.random.rand() < mutation_rate:
        noise = np.random.normal(0.0, mutation_strength, size=chromosome.shape)
        chromosome += noise
        np.clip(chromosome, 0.0, 1.0, out=chromosome)


def evolve_population(population):
    # Créer la génération suivante
    population.sort(key=fitness_score)

    # Garder les meilleures solutions (élitisme)
    next_generation = population[:num_elites]

    # Remplir le reste avec la descendance
    while len(next_generation) < population_size:
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)

        if np.random.rand() < crossover_prob:
            child1, child2 = crossover(parent1, parent2, blend_factor=0.25)
        else:
            child1, child2 = parent1.copy(), parent2.copy()

        mutate_chromosome(child1, mutation_prob)
        mutate_chromosome(child2, mutation_prob)
        next_generation.extend([child1, child2])

    return next_generation[:population_size]


# Commencer avec des solutions aléatoires
population = [random_chromosome() for _ in range(population_size)]
fitness_history = []


# Configuration des graphiques
fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])
layout_ax = fig.add_subplot(gs[0, 0])
fitness_ax = fig.add_subplot(gs[0, 1])

# Visualisation du layout principal
layout_ax.set_title("Meilleure solution actuelle")
layout_ax.set_xlim(0, BOX_WIDTH)
layout_ax.set_ylim(0, BOX_HEIGHT)
layout_ax.set_aspect('equal')
layout_ax.add_patch(Rectangle((0, 0), BOX_WIDTH, BOX_HEIGHT, fill=False, lw=2))

# Créer les patches rectangulaires pour l'animation
rectangle_patches = [
    Rectangle((0, 0), 1, 1, facecolor='lightblue',
              alpha=0.7, edgecolor='black')
    for _ in range(num_rects)
]
for patch in rectangle_patches:
    layout_ax.add_patch(patch)

# Graphique de progression du fitness
fitness_ax.set_title("Évolution du fitness")
fitness_ax.set_xlabel("Génération")
fitness_ax.set_ylabel("Coût total")
fitness_line, = fitness_ax.plot([], [], 'b-', linewidth=2)

# Texte d'information
info_text = layout_ax.text(0.02, 0.98, "", transform=layout_ax.transAxes,
                           verticalalignment="top", fontsize=10)


def update_animation(frame_num):
    # Appelée à chaque frame pour mettre à jour l'animation
    global population

    # Évoluer vers la génération suivante
    population = evolve_population(population)
    population.sort(key=fitness_score)

    # Récupérer la meilleure solution jusqu'à présent
    best_solution = population[0]
    best_cost = fitness_score(best_solution)
    fitness_history.append(best_cost)

    # Mettre à jour les positions des rectangles
    current_rects = decode_chromosome(best_solution)
    for patch, rect in zip(rectangle_patches, current_rects):
        x1, y1, x2, y2 = rect
        patch.set_xy((x1, y1))
        patch.set_width(x2 - x1)
        patch.set_height(y2 - y1)

    # Mettre à jour la courbe de fitness
    generation_numbers = np.arange(len(fitness_history))
    fitness_line.set_data(generation_numbers, fitness_history)
    fitness_ax.relim()
    fitness_ax.autoscale_view()

    # Mettre à jour l'affichage des infos
    info_text.set_text(
        f"Génération: {frame_num + 1}/{generations}\n"
        f"Meilleur coût: {best_cost:.2f}\n"
        f"Population: {population_size}\n"
        f"Croisement: {crossover_prob}, Mutation: {mutation_prob}"
    )

    return (*rectangle_patches, fitness_line, info_text)


# Lancer l'animation
animation = FuncAnimation(fig, update_animation, frames=generations,
                          interval=80, blit=False, repeat=False)

fig.suptitle(
    "Test d'un algo génétique", fontsize=14)
plt.tight_layout()
plt.show()
