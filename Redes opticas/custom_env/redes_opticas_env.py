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
        # Calcula la suma total del ancho de banda asignado en el ciclo actual.
        total_bandwidth_assigned = np.sum(self.band_onus)
        # Calcula la desviación promedio del ancho de banda asignado respecto al Balloc.
        average_deviation = np.mean(np.abs(self.band_onus - self.Balloc))
        # Calcula la capacidad restante del OLT.
        remaining_OLT_capacity = self.OLT_capacity - total_bandwidth_assigned

        info = {
            'total_bandwidth_assigned': total_bandwidth_assigned,
            'average_deviation_from_Balloc': average_deviation,
            'remaining_OLT_capacity': remaining_OLT_capacity,
        }
        return info
    

    def _calculate_reward(self):
        # Calcula la diferencia absoluta entre el Balloc y el ancho de banda asignado a cada ONU.
        deviation = np.abs(self.band_onus - self.Balloc)

        # La recompensa es negativa y proporcional a la desviación total de Balloc.
        # Esto significa que el agente es incentivado a minimizar la desviación.
        reward = -np.sum(deviation)

        return reward

    def reset(self, seed=None, options=None):

        #super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Restablece las asignaciones de ancho de banda a un estado inicial, por ejemplo, a cero o a un valor base.
        self.band_onus = np.zeros(self.num_onus, dtype=np.float32)
        # Restablece las asignaciones de ancho de banda previas también.
        self.previous_band_onus = np.zeros(self.num_onus, dtype=np.float32)
        # Define la capacidad total del OLT para este nuevo episodio, si es fijo o puede variar.
        self.OLT_capacity = self.Bmax * self.num_onus  # Ejemplo con capacidad máxima fija.
        
        # Genera una nueva demanda de ancho de banda para cada ONU si tu modelo lo requiere.
        # Esto puede depender de tu modelo específico y cómo quieres simular la variabilidad de la demanda.
        # Por ejemplo, si deseas comenzar cada episodio con una demanda aleatoria dentro de cierto rango:
        self.demand = np.random.uniform(low=0, high=self.Bmax, size=self.num_onus).astype(np.float32)

        # Puede que también quieras restablecer otras variables de estado aquí, dependiendo de tu modelo.

        # Devuelve el estado inicial observado.

        observation = self._get_obs()
        
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        # Actualizar las asignaciones de ancho de banda basadas en la acción tomada.
        # La acción será un vector con las asignaciones de ancho de banda deseadas para cada ONU.
        self.previous_band_onus = np.copy(self.band_onus)  # Guarda el estado anterior
        self.band_onus = np.clip(action, 0, self.Bmax)  # Asegura que las asignaciones estén dentro de los límites
        
        # Calcular la recompensa basada en cuán cerca están las asignaciones del Balloc
        reward = self._calculate_reward()

        # Actualizar el estado del entorno, como la demanda de cada ONU, si es aplicable.
        # Esto podría incluir variaciones en la demanda basadas en un modelo estocástico o en patrones determinados.
        # Por simplicidad, aquí solo simulamos un cambio aleatorio en la demanda.
        self.demand = np.random.uniform(low=0, high=self.Bmax, size=self.num_onus).astype(np.float32)

        # Determinar si el episodio ha terminado. En muchos entornos de redes, podría no haber
        # un "final" claro hasta que se alcance un número determinado de pasos, o podrías definir
        # condiciones específicas de terminación relacionadas con el rendimiento de la red.
        done = False  # Aquí simplemente continuamos indefinidamente o hasta un límite de pasos (no mostrado)
        
        # Opcional: Puede que quieras incluir información adicional para el debugging o análisis.
        info = self._get_info()

        # Asegurarse de devolver la observación del nuevo estado, la recompensa, si el episodio ha terminado,
        # y cualquier información adicional.
        return self._get_obs(), reward, done, False, info



from gymnasium.envs.registration import register

register(
    id="RedesOpticasEnv-v0",
    entry_point="custom_env.redes_opticas_env:RedesOpticasEnv",
    max_episode_steps=300,
)
