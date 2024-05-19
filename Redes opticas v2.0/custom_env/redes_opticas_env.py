import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
from scipy.stats import pareto

class RedesOpticasEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, num_ont=3, Vt=10e6, tamanoBuffer=20e6):
        self.num_ont = num_ont
        self.Vt = Vt  # bits por segundo (bps)
        self.tamanoBuffer = tamanoBuffer
        self.temp_ciclo = 0.002  # segundos (s)
        self.OLT_Capacity = Vt * self.temp_ciclo  # bits
        self.observation_space = spaces.Box(low=0, high=self.OLT_Capacity, shape=(self.num_ont,), dtype=np.float32)
        self.action_space = spaces.Box(low=-self.OLT_Capacity / 20, high=self.OLT_Capacity / 20, shape=(self.num_ont,), dtype=np.float32)

        self.on_off_state = False
        self.step_durations = []
        self.trafico_entrada = []
        self.trafico_pareto_futuro = []
        self.trafico_salida = []
        self.trafico_pareto_actual = []
        self.trafico_pendiente = np.zeros(self.num_ont)  # Inicializar el tráfico pendiente para cada ONT
        self.last_reward = 0
        self.tamano_cola = 0  # Inicializar tamaño de la cola como una variable int/float
        self.velocidadContratada = 0
        self.state = None
        self.reset()

    def _get_obs(self):
        obs = np.clip(self.trafico_entrada, 0, self.OLT_Capacity)
        return np.squeeze(obs)
    
    def _get_info(self):
        info = {
            'OLT_Capacity': self.OLT_Capacity,
            'trafico_entrada': self.trafico_entrada,
            'trafico_salida': self.trafico_salida,
            'trafico_IN_ON_actual': self.trafico_pareto_actual,
            'trafico_IN_ON_futuro': self.trafico_pareto_futuro,
            'tamano_cola': self.tamano_cola,
            'trafico_pendiente': self.trafico_pendiente
        }
        return info

    def calculate_pareto(self, num_ont=5, traf_pas=[]):
        alpha_ON = 2
        alpha_OFF = 1
        Vel_tx_max = self.Vt
        trafico_futuro_valores = []
        lista_trafico_act = []
        trafico_actual_lista = [[] for _ in range(self.num_ont)]

        for i in range(num_ont):
            if not traf_pas:
                trafico_pareto = list(np.random.pareto(alpha_ON, size=(1)))
                trafico_pareto += list(np.random.pareto(alpha_OFF, size=(1)))
            else:
                trafico_pareto = traf_pas[i]

            suma = sum(trafico_pareto)
            while suma < 2:
                trafico_pareto += list(np.random.pareto(alpha_ON, size=(1))) + list(np.random.pareto(alpha_OFF, size=(1)))
                suma = sum(trafico_pareto)

            traf_act = []
            suma = 0
            while suma < 2:
                traf_act.append(trafico_pareto.pop(0))
                suma = sum(traf_act)

            traf_fut = [0, 0]
            if len(traf_act) % 2 == 0:
                traf_fut[0] = 0
                traf_fut[1] = suma - 2
                traf_act[-1] -= traf_fut[1]
            else:
                traf_fut[0] = suma - 2
                traf_fut[1] = trafico_pareto[-1]
                traf_act[-1] -= traf_fut[0]

            trafico_actual_lista[i].append(traf_act)
            vol_traf_act = sum(traf_act[::2]) * Vel_tx_max * self.temp_ciclo
            lista_trafico_act.append(vol_traf_act)
            trafico_futuro_valores.append(traf_fut)

        return lista_trafico_act, trafico_actual_lista, trafico_futuro_valores

    def _calculate_reward(self):
        # Penalizar fuertemente el tamaño de la cola para mantenerlo lo más bajo posible
        reward = -self.tamano_cola * 100
        return reward

    def step(self, action):
        start_time = time.time()

        # Obtener el tráfico de entrada actual
        self.trafico_entrada, self.trafico_pareto_actual, self.trafico_pareto_futuro = self.calculate_pareto(self.num_ont, self.trafico_pareto_futuro)

        # Considerar el tráfico pendiente en el cálculo del tráfico de salida
        total_trafico = self.trafico_entrada + self.trafico_pendiente
        self.trafico_salida = np.clip(total_trafico + action, 0, self.OLT_Capacity)

        # Introducir una pequeña variabilidad en el tráfico de salida
        variabilidad = np.random.uniform(-0.05, 0.05, size=self.num_ont) * self.OLT_Capacity
        self.trafico_salida += variabilidad
        self.trafico_salida = np.clip(self.trafico_salida, 0, self.OLT_Capacity)

        # Asegurarse de que la suma del tráfico de salida no supere la capacidad del OLT
        if np.sum(self.trafico_salida) > self.OLT_Capacity:
            exceso = np.sum(self.trafico_salida) - self.OLT_Capacity
            self.trafico_salida -= (exceso / self.num_ont)  # Distribuir el exceso entre todas las ONTs
            self.trafico_salida = np.clip(self.trafico_salida, 0, self.OLT_Capacity)

        # Calcular el tráfico no transmitido para cada ONT
        self.trafico_pendiente = np.maximum(total_trafico - self.trafico_salida, 0)

        # Cálculo del tamaño de la cola acumulada
        self.tamano_cola += np.sum(self.trafico_pendiente)

        # Limitar el tamaño de la cola a la capacidad del Buffer
        self.tamano_cola = min(self.tamano_cola, self.tamanoBuffer)

        # Calcular recompensa
        reward = self._calculate_reward()

        # Acción correctiva para disminuir el tamaño de la cola
        if self.last_reward is not None and reward < self.last_reward:
            adjustment = np.random.uniform(0, 20, size=self.num_ont)
            self.trafico_salida = np.clip(self.trafico_salida + adjustment, 0, self.OLT_Capacity)

            # Asegurarse de que la suma del tráfico de salida no supere la capacidad del OLT nuevamente
            if np.sum(self.trafico_salida) > self.OLT_Capacity:
                exceso = np.sum(self.trafico_salida) - self.OLT_Capacity
                self.trafico_salida -= (exceso / self.num_ont)
                self.trafico_salida = np.clip(self.trafico_salida, 0, self.OLT_Capacity)

        self.last_reward = reward
        done = np.random.rand() > 0.99

        elapsed_time = time.time() - start_time
        if elapsed_time < 0.002:
            time.sleep(0.002 - elapsed_time)

        end_time = time.time()
        step_duration = end_time - start_time
        self.step_durations.append(step_duration)

        info = self._get_info()

        return self._get_obs(), reward, done, False, info

    def reset(self, seed=None, options=None):
        self.trafico_entrada, self.trafico_pareto_actual, self.trafico_pareto_futuro = self.calculate_pareto(self.num_ont, self.trafico_pareto_futuro)
        self.trafico_salida = np.random.uniform(low=0, high=self.OLT_Capacity, size=self.num_ont).astype(np.float32)
        self.tamano_cola = 0  # Reiniciar el tamaño de la cola como una variable int/float
        self.trafico_pendiente = np.zeros(self.num_ont)  # Reiniciar el tráfico pendiente para cada ONT
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def render(self, mode='human', close=False):
        if mode == 'human':
            pass

from gymnasium.envs.registration import register

register(
    id="RedesOpticasEnv-v0",
    entry_point="custom_env.redes_opticas_env:RedesOpticasEnv",
    max_episode_steps=300,
)
