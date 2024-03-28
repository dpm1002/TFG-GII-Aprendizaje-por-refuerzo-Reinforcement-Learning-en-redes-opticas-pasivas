import numpy as np

import gymnasium as gym
from gymnasium import spaces

class RedesOpticasEnv(gym.Env):
    """
    Entorno personalizado para controlar un entorno de redes opticas por refuerzo.
    El valor que intentaremos maximizar es el de que todas las ONUs se acerquen al valor
    de el BAlloc(ancho de banda que dictaminemos).
    """

    """
    Esta línea define las opciones de visualización para el entorno de simulación, 
    indicando que admite dos modos de renderización (human para visualización gráfica y rgb_array para obtener imágenes del estado) 
    y establece la tasa de actualización visual en 4 fotogramas por segundo.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, num_onus=10, Bmax=1000, Balloc=500):
        # Número de onus de la red
        self.num_onus = num_onus
        # Ancho de banda maximo de la red
        self.Bmax = Bmax
        # Ancho de banda que la red quiere dar a las onus
        self.Balloc = Balloc

        # El espacio de observación podría incluir información sobre:
        # - La demanda de ancho de banda actual de cada ONU
        # - El ancho de banda previamente asignado a cada ONU
        # - Información sobre la capacidad total del OLT
        self.observation_space = spaces.Dict({
            'band_onus': spaces.Box(low=0, high=self.Bmax, shape=(self.num_onus,), dtype=np.float32),
            'previous_band_onus': spaces.Box(low=0, high=self.Bmax, shape=(self.num_onus,), dtype=np.float32),
            'OLT_capacity': spaces.Box(low=0, high=self.Bmax*self.num_onus, shape=(1,), dtype=np.float32),
        })


        # El espacio de acción define las posibles asignaciones de ancho de banda que el agente puede hacer
        # Podría ser una asignación continua dentro de un rango, o un conjunto discreto de opciones
        self.action_space = spaces.Box(low=0, high=self.Bmax, shape=(self.num_onus,), dtype=np.float32)

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
        # CALCULA LA MEDIA DE TODOS LOS ANCHOS DE BANDA EN EL CICLO ACTUAL.
        total_mean_bandwidth_assigned = np.mean(self.band_onus)
        # Calcula la desviación promedio del ancho de banda asignado respecto al Balloc.
        average_deviation = np.mean(np.abs(self.band_onus - self.Balloc))
        # Calcula la capacidad restante del OLT.
        remaining_OLT_capacity = self.OLT_capacity - total_mean_bandwidth_assigned

        info = {
            'total_mean_bandwidth_assigned': total_mean_bandwidth_assigned,
            'average_deviation_from_Balloc': average_deviation,
            'remaining_OLT_capacity': remaining_OLT_capacity,
        }
        return info
    

    def _calculate_reward(self):
        # La recompensa es la suma del ancho de banda asignado a cada ONU.
        # Esto incentivará al agente a aumentar el ancho de banda asignado.
        reward = np.mean(self.band_onus)

        return reward

    def reset(self, seed=None, options=None):
        # Puede ser útil llamar a super().reset() si estás heredando de una clase que ya implementa un método reset
        # Sin embargo, si no es necesario, puedes omitirlo o comentarlo si super() no tiene un método reset
        # super().reset(seed=seed)

        # Si decides establecer una semilla para la reproducibilidad (opcional)
        if seed is not None:
            np.random.seed(seed)

        # Asignaciones de ancho de banda aleatorias iniciales o basadas en algún criterio
        self.band_onus = np.random.uniform(low=0, high=self.Bmax, size=self.num_onus).astype(np.float32)
        
        # Para las asignaciones de ancho de banda anteriores, puedes elegir mantenerlas en cero o darles un valor aleatorio
        self.previous_band_onus = np.zeros(self.num_onus, dtype=np.float32)
        
        # Capacidad del OLT podría ser constante o podrías introducir alguna variabilidad
        self.OLT_capacity = np.random.uniform(low=self.Bmax * 0.8, high=self.Bmax, size=1).astype(np.float32)
        
        # Demanda de ancho de banda que varía cada episodio
        self.demand = np.random.uniform(low=0, high=self.Bmax, size=self.num_onus).astype(np.float32)

        # Obtén el estado inicial y la información adicional
        observation = self._get_obs()
        info = self._get_info()

        # Devuelve el estado inicial observado y cualquier información adicional
        return observation, info
    
    def step(self, action):
        # Guarda el estado anterior
        self.previous_band_onus = np.copy(self.band_onus)

        # Asegura que las asignaciones estén dentro de los límites y puedan cambiar significativamente
        self.band_onus += np.clip(action, -self.Bmax/10, self.Bmax/10)
        self.band_onus = np.clip(self.band_onus, 0, self.Bmax)
        
        # Recompensa basada en la desviación de Balloc
        reward = self._calculate_reward()

        # Simulación de la variabilidad en la demanda
        demand_change = np.random.uniform(low=-10, high=10, size=self.num_onus)
        self.demand = np.clip(self.demand + demand_change, 0, self.Bmax)
        
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
