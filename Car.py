import pygame
import numpy as np
import math
import logging
from sensor import RayCaster  

# Fonction de test de collision
def collision_test(rect, obstacles, screen_width=450, screen_height=700):
    """
    Vérifie les collisions entre une entité et des obstacles ou si elle sort des limites de l'écran.
    """
    if rect.left < 0 or rect.right > screen_width or rect.top < 0 or rect.bottom > screen_height:
        return True
    return any(rect.colliderect(obstacle.rect) for obstacle in obstacles)

# Classe de base pour les entités
class Entity:
    def __init__(self, img, xy, size=None):
        self.xy = list(xy)
        self.size = size if size else (img.get_width(), img.get_height())
        self.surface = pygame.transform.scale(img, self.size) if size else img
        self.rect = pygame.Rect(self.xy[0], self.xy[1], self.size[0], self.size[1])

    def display(self, surface):
        surface.blit(self.surface, self.rect.topleft)

# Classe pour la voiture principale
class Car(Entity):
    def __init__(self, img, xy, size=None, speed=0, angle=0, acceleration=2, friction=1, agent=None, max_speed=16, distance_traveled=0):
        super().__init__(img, xy, size)
        self.angle = angle
        self.speed = speed
        self.agent = agent
        self.acceleration = acceleration
        self.friction = friction
        self.max_speed = max_speed
        self.damaged = False
        self.sensor = RayCaster(160)
        self.alerts = [0] * self.sensor.nor
        self.time_since_last_damage = 0  # Temps écoulé depuis la dernière collision
        self.distance_traveled = distance_traveled  # Distance totale parcourue

    def calculate_reward(self):
        reward = 0

        # Pénalité pour collision
        if self.damaged:
            return -10

        # Récompense pour éviter les collisions
        if not self.damaged:
            self.time_since_last_damage += 1
            if self.time_since_last_damage > 100:
                reward += 2
        else:
            self.time_since_last_damage = 0

        # Récompense pour la vitesse
        if self.speed > 0:
            reward += min(self.speed / self.max_speed, 1)
        else:
            reward -= 1

        # Récompense pour la distance parcourue
        reward += min(self.distance_traveled / 100, 5)
        return reward

    def get_state(self):
        return np.array([self.rect.x, self.rect.y, self.speed, self.angle, *self.alerts])

    def update(self, obstacles):
        if self.damaged:
            logging.info("La voiture est endommagée. Réinitialisation requise.")

        # Obtenir l'état actuel et l'action à effectuer
        state = self.get_state()
        action = self.agent.act(state)

        # Actions possibles
        directions = [0, 0, 0]
        if action == 0:
            directions[0] = 1  # Tourner à gauche
        elif action == 1:
            directions[1] = 1  # Avancer
        elif action == 2:
            directions[2] = 1  # Tourner à droite

        # Gestion de la vitesse
        if directions[1]:
            self.speed = min(self.speed + self.acceleration, self.max_speed)
        else:
            self.speed = max(self.speed - self.friction, 0)

        # Gestion de l'angle
        if directions[0]:
            self.angle += 3
        if directions[2]:
            self.angle -= 3
        self.angle %= 360

        # Mise à jour de la position
        self.rect.x -= math.sin(math.radians(self.angle)) * self.speed
        self.distance_traveled += math.sqrt(
            math.sin(math.radians(self.angle))**2 * self.speed**2 +
            math.cos(math.radians(self.angle))**2 * self.speed**2
        )

        # Vérification des collisions
        self.damaged = collision_test(self.rect, obstacles)
        return action

    def cast_rays(self, surface, obstacles, display_rays=True):
        self.alerts = self.sensor.detect(self.rect.center, self.angle, obstacles)
        if display_rays:
            self.sensor.display(surface, self.rect.center)

    def reset(self, xy, speed=0, angle=0):
        self.rect.x, self.rect.y = xy
        self.angle = angle
        self.speed = speed
        self.damaged = False

    def display(self, surface, alpha=255):
        rotated_surface = pygame.transform.rotate(self.surface, self.angle)
        rotated_surface.set_alpha(alpha)
        self.rect = surface.blit(rotated_surface, (self.rect.x, self.rect.y))

# Classe pour les voitures de trafic
class TrafficCar(Entity):
    def __init__(self, img, position, speed=7, size=None):
        super().__init__(img, position, size)
        self.speed = speed

    def update(self):
        self.xy[1] += self.speed
        self.rect.y += self.speed
        if self.xy[1] > 700:
            self.reset()

    def reset(self):
        self.xy[1] = -100
        self.rect.y = -100
