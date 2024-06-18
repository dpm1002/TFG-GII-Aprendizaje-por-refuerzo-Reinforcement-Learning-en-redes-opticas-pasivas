import numpy as np

import gymnasium as gym
from gymnasium import spaces

class RedesOpticasEnv(gym.Env):
    """
    Entorno personalizado para controlar un entorno de redes opticas por refuerzo.
    El valor que intentaremos maximizar es el de que todas las ONUs se acerquen al valor
    de el bgarantizado(ancho de banda que dictaminemos).
    """

    """
    Esta línea define las opciones de visualización para el entorno de simulación, 
    indicando que admite dos modos de renderización (human para visualización gráfica y rgb_array para obtener imágenes del estado) 
    y establece la tasa de actualización visual en 4 fotogramas por segundo.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None,seed=0, num_onus=10, bmax=1000, bgarantizado=500):
        # Número de onus de la red
        self.num_onus = num_onus
        # Ancho de banda maximo de la red
        self.bmax = bmax
        # Ancho de banda que la red quiere dar a las onus
        self.bgarantizado = bgarantizado

        # Recompensa anterior
        self.last_reward=None

        # El espacio de observación incluye la información sobre:
        # - La demanda de ancho de banda actual de cada ONU
        # - El ancho de banda previamente asignado a cada ONU
        # - Información sobre la capacidad total del OLT
        self.observation_space = spaces.Dict({
            'band_onus': spaces.Box(low=0, high=self.bmax, shape=(self.num_onus,), dtype=np.float32),
            'previous_band_onus': spaces.Box(low=0, high=self.bmax, shape=(self.num_onus,), dtype=np.float32),
            'OLT_capacity': spaces.Box(low=0, high=self.bmax*self.num_onus, shape=(1,), dtype=np.float32),
        })


        # El espacio de acción define las posibles asignaciones de ancho de banda que el agente puede hacer
        # Podría ser una asignación continua dentro de un rango, o un conjunto discreto de opciones
        self.action_space = spaces.Box(low=0, high=self.bmax, shape=(self.num_onus,), dtype=np.float32)

        self.rng = np.random.default_rng(seed)  # Inicializa el generador de números aleatorios

        # Inicialización del estado
        self.state = None
        self.reset()
    
    def _get_obs(self):
        # Este método devuelve el estado actual como una observación.
        obs = {
            # 'band_onus' es un vector que representa el ancho de banda actualmente asignado a cada ONU.
            'band_onus': np.array(self.band_onus, dtype=np.float32),

            # 'previous_band_onus' es un vector que representa el ancho de banda asignado a cada ONU en el ciclo anterior.
            'previous_band_onus': np.array(self.previous_band_onus, dtype=np.float32),

            # 'OLT_capacity' es el ancho de banda total que el OLT puede distribuir entre todas las ONUs.
            'OLT_capacity': np.array([self.OLT_capacity], dtype=np.float32),
        }
        return obs
    
    def _get_info(self):
        # Saca la informacion del ciclo actual del ancho de banda de las onus
        band_onus=self.band_onus
        # CALCULA LA MEDIA DE TODOS LOS ANCHOS DE BANDA EN EL CICLO ACTUAL.
        total_mean_bandwidth_assigned = np.mean(self.band_onus)
        # Calcula la desviación promedio del ancho de banda asignado respecto al bgarantizado.
        average_deviation = np.mean(np.abs(self.band_onus - self.bgarantizado))
        # Calcula la capacidad restante del OLT.
        remaining_OLT_capacity = self.OLT_capacity - total_mean_bandwidth_assigned

        info = {
            'band_onus': band_onus,
            'total_mean_bandwidth_assigned': total_mean_bandwidth_assigned,
            'average_deviation_from_bgarantizado': average_deviation,
            'remaining_OLT_capacity': remaining_OLT_capacity,
        }
        return info
    

    def _calculate_reward(self):
        # Calcula la desviación media de la asignación de ancho de banda de cada ONU del valor objetivo bgarantizado.
        average_deviation = np.mean(np.abs(self.band_onus - self.bgarantizado))
        # La recompensa es inversamente proporcional a la desviación media.
        # Penalizaciones más altas para desviaciones medias mayores.
        # Esto significa que cuanto mayor sea la desviación promedio (cuanto peor sea el desempeño), más negativa será la recompensa. 
        reward = -average_deviation
        return reward

    def reset(self, seed=None, options=None):

        # Asignaciones de ancho de banda aleatorias iniciales 
        self.band_onus = self.rng.uniform(low=0, high=self.bmax, size=self.num_onus).astype(np.float32)
        
        # Para las asignaciones de ancho de banda anteriores las mantenemos a 0
        self.previous_band_onus = np.zeros(self.num_onus, dtype=np.float32)
        
        # Capacidad del OLT la mantenemos uniforme
        self.OLT_capacity = self.rng.uniform(low=self.bmax * 0.8, high=self.bmax, size=1).astype(np.float32)
        
        # Demanda de ancho de banda que varía cada episodio
        self.demand = self.rng.uniform(low=0, high=self.bmax, size=self.num_onus).astype(np.float32)

        # Obtenemos el estado inicial y la información adicional
        observation = self._get_obs()
        info = self._get_info()

        # Devolvemos el estado inicial observado y cualquier información adicional
        return observation, info
    
    def step(self, action):
        # Guardamos el estado anterior de los valores del ancho de banda
        self.previous_band_onus = np.copy(self.band_onus)

        # Aseguramos que las asignaciones estén dentro de los límites y puedan cambiar significativamente
        self.band_onus += np.clip(action, -self.bmax/10, self.bmax/10)
        self.band_onus = np.clip(self.band_onus, 0, self.bmax)
        
        # Recompensa basada en la desviación de bgarantizado
        reward = self._calculate_reward()

        # Ponemos condiciones al reward para que estimule la mejora.
        if self.last_reward is not None and reward < -self.last_reward:
            # Este ajuste es un ejemplo y podría no ser la mejor solución
            adjustment = self.rng.uniform(0, 20)
            self.band_onus = np.where(self.band_onus < self.bgarantizado, self.band_onus + adjustment, self.band_onus - adjustment)
            self.band_onus = np.clip(self.band_onus, 0, self.bmax)
            
        self.last_reward = reward

        # Simulación de la variabilidad en la demanda
        demand_change = self.rng.uniform(low=-10, high=10, size=self.num_onus)
        self.demand = np.clip(self.demand + demand_change, 0, self.bmax)
        
        # Determinación de si el episodio ha terminado
        done = np.random.rand() > 0.99  # Ejemplo: termina el 1% de las veces al azar
        
        # Recopilación de información adicional para análisis
        info = self._get_info()

        # Devolución de la observación, recompensa, finalización del episodio y cualquier información adicional
        return self._get_obs(), reward, done, False, info


from gymnasium.envs.registration import register

register(
    id="RedesOpticasEnv-v0",
    entry_point="custom_env.redes_opticas_env:RedesOpticasEnv",
    max_episode_steps=300,
)
