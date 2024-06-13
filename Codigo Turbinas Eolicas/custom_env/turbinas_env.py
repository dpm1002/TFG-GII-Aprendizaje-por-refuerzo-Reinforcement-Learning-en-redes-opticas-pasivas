import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class TurbinasEnv(gym.Env):
    """
    Entorno personalizado para controlar un parque eólico con aprendizaje por refuerzo.
    Cada turbina se puede orientar para maximizar la producción total de energía en función de las condiciones del viento.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, num_turbines=5, max_wind_speed=25):
        # Número de turbinas en el parque eólico
        self.num_turbines = num_turbines
        # Velocidad máxima del viento en m/s para la simulación
        self.max_wind_speed = max_wind_speed

        # Las observaciones incluyen la velocidad del viento, la dirección del viento y la orientación para cada turbina.
        # La velocidad del viento es una variable continua, la dirección del viento es un ángulo y la orientación también es un ángulo.
        # Asumimos que la velocidad y dirección del viento son las mismas para todo el parque por simplicidad.
        self.observation_space = spaces.Dict(
            {
                "wind_speed": spaces.Box(low=0, high=max_wind_speed, shape=(1,), dtype=np.float32),
                "wind_direction": spaces.Box(low=0, high=2*np.pi, shape=(1,), dtype=np.float32),
                "turbine_orientations": spaces.Box(low=0, high=2*np.pi, shape=(num_turbines,), dtype=np.float32)
            }
        )

        # Las acciones pueden ser rotar cada turbina por un cierto ángulo dentro de [-pi, pi].
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(num_turbines,), dtype=np.float32)

        # Verificación del modo de renderizado
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        # Si se utiliza la representación en modo humano, `self.window` será una referencia
        # a la ventana donde dibujamos, y `self.clock` será un reloj que se utiliza
        # para asegurar que el entorno se renderiza con la frecuencia de cuadros correcta.
        self.window = None
        self.clock = None

        # Inicialización del estado
        self.state = None
        self.reset()
    
    def _get_obs(self):
        return {
        "wind_speed": np.array([self.wind_speed], dtype=np.float32),
        "wind_direction": np.array([self.wind_direction], dtype=np.float32),
        "turbine_orientations": np.array(self.turbine_orientations, dtype=np.float32)
    }
    
    def _get_info(self):
        # Calcula la potencia total generada por el parque eólico
        # Aquí asumimos que todas las turbinas tienen la misma longitud de pala, que deberías definir o calcular
        blade_length = 50 # Longitud de la pala, 50 metros
        total_power = sum(self._calculate_power(orientation, self.wind_speed, self.wind_direction, blade_length)
                        for orientation in self.turbine_orientations)
        return {"total_power": total_power}

    def _calculate_power(self, turbine_orientation, wind_speed, wind_direction, blade_length):
        # Densidad del aire
        rho = 1.225  # kg/m^3

        # Área barrida por las palas de la turbina (A = πR^2)
        A = np.pi * blade_length**2

        # Ajusta la velocidad del viento por el coseno del ángulo entre la dirección del viento y la orientación de la turbina
        # Asumimos que la potencia es 0 si el viento está soplando desde atrás
        relative_wind_speed = max(wind_speed * np.cos(wind_direction - turbine_orientation), 0)

        # Calcula la potencia producida por una turbina individual usando la fórmula proporcionada
        power = 0.5 * A * rho * relative_wind_speed**3
        return power

    def reset(self, seed=None, options=None):
        # Se establece la semilla para la generación de números aleatorios
        super().reset(seed=seed)

        # Inicializar la velocidad y dirección del viento con valores aleatorios dentro de los rangos permitidos
        
        #Comento para hacer pruebas
        self.wind_speed = self.np_random.uniform(low=0, high=self.max_wind_speed)
        #self.wind_speed=25
        
        #Comento para hacer pruebas
        self.wind_direction = self.np_random.uniform(low=0, high=2*np.pi)
        #self.wind_direction = 5.0

        # Inicializar la orientación de cada turbina con valores aleatorios o una orientación fija, dependiendo de la simulación
        self.turbine_orientations = self.np_random.uniform(low=0, high=2*np.pi, size=(self.num_turbines,))

        # Crear la observación inicial basada en los valores iniciales
        observation = self._get_obs()

        # Si es necesario, obtener información adicional (puede ser útil para debugging o aprendizaje)
        info = self._get_info()

        # Si el modo de renderizado está activo, renderizar el entorno
        #if self.render_mode == "human":
            #self.render()

        return observation, info

    
    def step(self, action):
        # Aplica la acción a la orientación de las turbinas
        # Asegúrate de que las nuevas orientaciones estén dentro del rango [0, 2*pi]
        self.turbine_orientations = (self.turbine_orientations + action) % (2 * np.pi)

        # Simula el cambio en la velocidad del viento
        # Por ejemplo, puede variar aleatoriamente dentro de un rango determinado o puede tener una tendencia
        
        wind_speed_change = self.np_random.uniform(-1, 1)  # Cambio de velocidad aleatorio entre -1 y 1 m/s
        self.wind_speed = np.clip(self.wind_speed + wind_speed_change, 0, self.max_wind_speed)

        #Voy a poner el viento estatico a una velocidad para pruebas

        #self.wind_speed=25

        #Cambio que no haya cambio en el aire

        # Simula el cambio en la dirección del viento
        # Este podría ser un cambio aleatorio o podría depender de algún otro factor
        wind_direction_change = self.np_random.uniform(-0.1, 0.1)  # Cambio de dirección aleatorio entre -0.1 y 0.1 radianes
        self.wind_direction = (self.wind_direction + wind_direction_change) % (2 * np.pi)

        #self.wind_direction=5.0

        # Calcula la recompensa basada en la potencia total generada
        reward = self._calculate_total_reward()

        # Actualiza la observación con el nuevo estado
        observation = self._get_obs()

        # Información adicional opcional
        info = self._get_info()

        # En un entorno real, podrías tener una condición de terminación (por ejemplo, tiempo máximo alcanzado)
        terminated = False  # Por ahora, no tenemos una condición de terminación

        # Si se está utilizando renderizado, actualiza la visualización del entorno
        #if self.render_mode == "human":
        #    self.render()

        return observation, reward, terminated, False, info

    def _calculate_total_reward(self):
        # Calcula la recompensa total, que podría ser la suma de la potencia generada por todas las turbinas
        blade_length=50
        total_power = sum(self._calculate_power(orientation, self.wind_speed, self.wind_direction, blade_length)
                        for orientation in self.turbine_orientations)
        # La recompensa podría ser simplemente la potencia total generada
        reward = total_power
        return reward




from gymnasium.envs.registration import register

register(
    id="TurbinasEnv-v0",
    entry_point="custom_env.turbinas_env:TurbinasEnv",
    max_episode_steps=300,
)