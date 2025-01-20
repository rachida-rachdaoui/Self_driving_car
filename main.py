import os
import time
import pygame               
import torch
import streamlit as st       ###  streamlit run main.py  ###    pour executer le fichier avec le mode streamlit
from random import choice
from Car import TrafficCar, Car
from ddqn import DDQNAgent

# Constantes globales
BASE_PATH = os.path.dirname(__file__)
SCREEN_SIZE = (450, 600)
FPS = 30
MAX_TRAFFIC_CARS = 2
TRAFFIC_GENERATION_INTERVAL = 3
BATCH_SIZE = 32
MEMORY_THRESHOLD = BATCH_SIZE * 10
TRAINING_INTERVAL = 5
TARGET_UPDATE_INTERVAL = 10

# Chargement des images
TRAFFIC_IMG = [
    pygame.image.load(os.path.join(BASE_PATH, f'./assets/vehicles/{img}'))
    for img in ["orange.png", "red.png", "truck.png", "white.png", "cyan.png"]
]
TRAFFIC_XY = [[(350, -100), (200, -100), (50, -100)]]
AGENT_IMG = pygame.image.load(os.path.join(BASE_PATH, './assets/vehicles/yellow.png'))

# Initialisation de Pygame
pygame.init()
SCREEN = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption('Autodriven')
clock = pygame.time.Clock()

# Initialisation des agents
npl = [9, 128, 128]
n_actions = 3
ddqn_agent = DDQNAgent(alpha=0.001, gamma=0.998, n_actions=n_actions, epsilon=1.0, batch_size=128, input_dims=9)
agent_car = Car(AGENT_IMG, (200, 500), size=(50, 100), agent=ddqn_agent)

# Variables principales
traffic = []
running = True
episode = 1
max_episodes = 1000
step = 1
total_reward = 0
rewards_list = []
last_traffic_time = time.time()

# Streamlit configuration
st.title("DQN Simulation - Rewards")
reward_chart = st.line_chart()

# Fonctions auxiliaires
def generate_traffic(traffic, max_cars, traffic_images, traffic_positions):
    if len(traffic) < max_cars:
        new_row = choice(traffic_positions)
        for place in new_row:
            if not any(car.rect.colliderect(pygame.Rect(place, (50, 100))) for car in traffic):
                traffic.append(TrafficCar(choice(traffic_images), place, speed=10, size=(50, 100)))

def remove_offscreen_traffic(traffic):
    return [car for car in traffic if car.rect.top <= SCREEN_SIZE[1]]

def mark_lanes(surface, color, start, end, length, gap):
    center_x = surface.get_width() // 2
    lane_width = 20
    left_lane = center_x - lane_width - 50
    right_lane = center_x + lane_width + 50

    for i in range(start[1], end[1], length + gap):
        pygame.draw.line(surface, color, (left_lane, i), (left_lane, i + length))
        pygame.draw.line(surface, color, (right_lane, i), (right_lane, i + length))

def load_model(agent, filename="dqn_model.pth"):
    if os.path.exists(filename):
        agent.load_state_dict(torch.load(filename))
        agent.eval()
        print(f"Modèle chargé depuis {filename}")
    else:
        print(f"Le fichier {filename} n'existe pas.")

# Boucle principale
while running and episode <= max_episodes:
    SCREEN.fill((50, 50, 50))
    done = False

    # Dessiner les lignes de voie
    mark_lanes(SCREEN, (255, 255, 0), (200, -100), (200, 800), 20, 10)

    # Réinitialisation en cas de collision
    if agent_car.damaged:
        rewards_list.append(total_reward)
        reward_chart.line_chart(rewards_list)
        total_reward = 0
        done = True
        agent_car.reset((200, 500), speed=0)
        traffic = []
        episode += 1
        step = 1
        continue

    # Sauvegarde périodique
    if episode % 500 == 0 and step == 1:
        ddqn_agent.save_model()

    # Gestion des événements
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Génération de trafic
    current_time = time.time()
    if current_time - last_traffic_time >= TRAFFIC_GENERATION_INTERVAL:
        generate_traffic(traffic, MAX_TRAFFIC_CARS, TRAFFIC_IMG, TRAFFIC_XY)
        last_traffic_time = current_time

    traffic = remove_offscreen_traffic(traffic)

    for car in traffic:
        car.update()
        car.display(SCREEN)

    # Mise à jour de l'agent
    state = agent_car.get_state()
    action = agent_car.update(obstacles=traffic)
    agent_car.cast_rays(SCREEN, [pygame.Rect(90, -100, 10, 800), pygame.Rect(400, -100, 10, 800)] + [car.rect for car in traffic])
    next_state = agent_car.get_state()
    reward = agent_car.calculate_reward()
    total_reward += reward

    ddqn_agent.remember(state, action, reward, next_state, done)
    ddqn_agent.learn()

    agent_car.display(SCREEN)
    step += 1
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
