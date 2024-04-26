import numpy as np

import gymnasium as gym
from gymnasium import spaces

from scipy.stats import pareto

class RedesOpticasEnv(gym.Env):

    """
    Esta línea define las opciones de visualización para el entorno de simulación, 
    indicando que admite dos modos de renderización (human para visualización gráfica y rgb_array para obtener imágenes del estado) 
    y establece la tasa de actualización visual en 4 fotogramas por segundo.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, num_ont=3, Vt=10e6):
        #Numero de ONTs de la red
        self.num_ont=num_ont

        #Tiempo de cada ciclo: 2ms
        self.temp_ciclo=0.002

        #Capacidad maxima del OLT
        self.OLT_Capacity=Vt*self.temp_ciclo

        #El espacio de observacion sera el de la capacidad de cada ONT, definido con un array
        #del valor de cada una de las capacidades
        self.observation_space=spaces.Box(low=0, high=self.OLT_Capacity, shape=(self.num_ont,), dtype=np.float32)

        #Definimos el espacio de acciones, en nuestro caso es el numero de ONTs sobre las 
        #que trabajaremos
        self.action_space = spaces.Box(low=-self.OLT_Capacity / 10, high=self.OLT_Capacity / 10, shape=(self.num_ont,), dtype=np.float32)

        #Registro de los estados de ON y OFF
        self.on_off_state=False

        # Inicialización del estado
        self.state = None
        self.reset()

    def _get_obs(self):
        # Asegurar que el estado siempre está dentro de los límites definidos por el espacio de observaciones
        obs = np.clip(self.state, 0, self.OLT_Capacity)
        return obs
    
    def _get_info(self):

        info={
            'OLT_Capacity':self.OLT_Capacity,
            'band_onts':self.state,
            'on_off_states':self.on_off_state
        }

        return info

    def step(self, action):
        is_on = np.random.choice([1, 0], p=[0.5, 0.5], size=self.num_ont)  # 1 para ON, 0 para OFF
        self.on_off_state = is_on  # Guardar el estado actual ON/OFF

        # Base de variación de tráfico
        base_traffic_variation = pareto.rvs(1.5, size=self.num_ont) * self.OLT_Capacity / 100
        
        # Genera un pico de variación de tráfico con cierta probabilidad, esto es para valores mas realistas
        peak_trigger = np.random.rand()  # Valor entre 0 y 1
        peak_factor = 10 if peak_trigger > 0.75 else 1  # Pico aleatorio (10x) si el valor es mayor a 0.75

        # Calcula la variación de tráfico con la posibilidad de un pico
        traffic_variation = is_on * base_traffic_variation * peak_factor  # Pico si el trigger es alto
        
        # Actualiza el estado con la variación de tráfico, considerando el pico
        self.state = np.clip(self.state + action + traffic_variation, 0, self.OLT_Capacity)
        
        # Recompensa para mantener el tráfico bajo control, la basamos en que el trafico este regulado a un mismo ancho de banda.
        reward = -np.sum(self.state)

        # Ajusta el estado si excede la capacidad del OLT
        while np.sum(self.state) > self.OLT_Capacity:
            self.state = self.state * (self.OLT_Capacity / np.sum(self.state))
        
        # Determinación de si el episodio ha terminado
        done = np.random.rand() > 0.99

        # Información adicional
        info = self._get_info()

        return self._get_obs(), reward, done, False, info
        

    def reset(self,seed=None, options=None):
        # Inicializa el estado usando una distribución de Pareto escalada a la capacidad máxima del OLT
        # Inicializa el estado usando una distribución de Pareto
        self.state = pareto.rvs(1.5, size=self.num_ont) * self.OLT_Capacity / 10
        # Obtenemos el estado inicial y la información adicional
        
        self.on_off_state = np.zeros(self.num_ont)  # Resetear también el registro de ON/OFF

        #Definimos los valores de observacion y de info
        observation = self._get_obs()
        info = self._get_info()
        return observation, info 


    def render(self, mode='human', close=False):
        if mode == 'human':
            print(f"Estado actual del tráfico: {self.state}")

from gymnasium.envs.registration import register

register(
    id="RedesOpticasEnv-v0",
    entry_point="custom_env.redes_opticas_env:RedesOpticasEnv",
    max_episode_steps=300,
)