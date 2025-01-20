import pygame
import numpy as np

# Définir les couleurs
SAFE = (0, 204, 153, 10)  # Vert clair (safe)
DANGER = (204, 0, 102, 10)  # Rouge clair (danger)
MAX_DETECTION_DISTANCE = 200  # Distance maximale de détection

# Calcul de la distance entre deux points
distance = lambda p1, p2: np.linalg.norm(np.array(p1) - np.array(p2))

class RayCaster:
    def __init__(self, coverage):
        """
        Initialisation des paramètres des capteurs (rayons).
        - coverage : portée maximale des rayons
        """
        self.nor = 5  # Nombre de rayons
        self.coverage = coverage  # Portée des rayons
        self.alerts = [0.0] * self.nor  # Liste des alertes (0 : pas de danger)
        self.angles = [ -60, -80, -90, -100, -120]  # Angles symétriques autour de 0°
        self.reach = [(0.0, 0.0)] * self.nor  # Positions atteintes par chaque rayon

    def detect(self, position, angle, obstacles):
        """
        Détecte les obstacles en lançant les rayons à partir de la position et de l'angle donnés.
        """
        self.alerts = [0.0] * self.nor  # Réinitialiser les alertes
        for i, sensor_angle in enumerate(self.angles):
            # Inverser le sens des angles pour s'adapter à Pygame
            # normalized_angle = (-sensor_angle + angle) % 360
            x = np.cos(np.radians(sensor_angle - angle)) * self.coverage + position[0]
            y = np.sin(np.radians(sensor_angle - angle)) * self.coverage + position[1]
            self.reach[i] = (x,y)

            # Détecter les intersections avec les obstacles
            for obstacle in obstacles:
                intercept = self.check_and_get_intersection(position, (x, y), obstacle)
                if intercept:
                    dist_to_intercept = distance(position, intercept)
                    if dist_to_intercept < self.coverage:
                        self.alerts[i] = (self.coverage - dist_to_intercept) / self.coverage
                        self.reach[i] = intercept  # Mettre à jour la portée atteinte
                        break  # Arrêter dès qu'une intersection est trouvée

        return self.alerts

    def check_and_get_intersection(self, start, end, obstacle):
        """
        Vérifie si un rayon intersecte un obstacle et calcule l'intersection.
        """
        if hasattr(obstacle, 'rect'):
            obstacle_rect = obstacle.rect
        elif isinstance(obstacle, pygame.Rect):
            obstacle_rect = obstacle
        else:
            raise ValueError("Obstacle must be a pygame.Rect or have a 'rect' attribute.")

        clipped_line = obstacle_rect.clipline(start, end)
        return clipped_line[0] if clipped_line else None

    def display(self, surface, position):
        """
        Affiche les rayons sur l'écran avec des couleurs selon la détection d'obstacles.
        """
        for i, alert in enumerate(self.alerts):
            color = SAFE if alert == 0.0 else DANGER
            pygame.draw.line(surface, color, position, self.reach[i])
