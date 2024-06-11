
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
from scipy.stats import pareto

class RedesOpticasEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, num_ont=3, Vt=10e6, Vt_contratada=10e6/10):
        self.num_ont = num_ont #numero de ont(unidades opticas)
        self.Vt = Vt  # bits por segundo (bps)
        self.temp_ciclo = 0.002  # segundos (s)
        self.OLT_Capacity = Vt * self.temp_ciclo  # bits
        #Velocidad de transmision contratada, lo ponemos a 1/10 del Vt de la OLT inicialmente
        self.velocidadContratada = Vt_contratada
        #Maximo de bits que se pueden transmitir en un ciclo en cada ont por la limitacion de la velocidad contratada
        self.Max_bits_ONT=self.velocidadContratada*self.temp_ciclo

        self.observation_space = spaces.Box(low=0, high=self.Max_bits_ONT, shape=(self.num_ont,), dtype=np.float32)
        self.action_space = spaces.Box(low=-self.Max_bits_ONT, high=self.Max_bits_ONT, shape=(self.num_ont,), dtype=np.float32)

        self.step_durations = []  #Guardar duracion de tiempo del ciclo
        self.trafico_entrada = []  #Guardar el trafico de entrada en cada ont
        self.trafico_pareto_futuro = []  #Guardar el trafico_pareto_futuro
        self.trafico_salida = []   #Guardar el trafico de salida en cada ont
        self.trafico_pareto_actual = []  #Guardar el trafico pareto actual
        self.trafico_pendiente = np.zeros(self.num_ont)  # Inicializar el tráfico pendiente para cada ONT
        
        self.state = None
        self.reset()

    def _get_obs(self):
        obs = np.clip(self.trafico_entrada, 0, self.OLT_Capacity) / self.Max_bits_ONT
        return np.squeeze(obs)
    
    def _get_info(self):
        info = {
            'OLT_Capacity': self.OLT_Capacity,
            'trafico_entrada': self.trafico_entrada,
            'trafico_salida': self.trafico_salida,
            'trafico_IN_ON_actual': self.trafico_pareto_actual,
            'trafico_pendiente': self.trafico_pendiente
        }
        return info

    def calculate_pareto(self, num_ont=5, traf_pas=[]):
        alpha_ON = 1.4
        alpha_OFF = 1.2
        Vel_tx_max = self.Vt*0.1
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
            vol_traf_act = sum(traf_act[::2]) * Vel_tx_max * 10e-3
            lista_trafico_act.append(vol_traf_act)
            trafico_futuro_valores.append(traf_fut)

        return lista_trafico_act, trafico_actual_lista, trafico_futuro_valores

    def _calculate_reward(self):
        # Penalizar fuertemente el tamaño de la cola para mantenerlo lo más bajo posible
        reward = -sum(self.trafico_pendiente)
        return reward

    def step(self, action):
        # print(action)
        start_time = time.time()

        # Obtener el tráfico de entrada actual
        self.trafico_entrada, self.trafico_pareto_actual, self.trafico_pareto_futuro = self.calculate_pareto(self.num_ont, self.trafico_pareto_futuro)

        # Considerar el tráfico pendiente en el cálculo del tráfico de salida
        self.trafico_salida = np.clip(action, 0, self.Max_bits_ONT)

        # Asegurar que si hay tráfico pendiente, se ajuste adecuadamente el tráfico de salida
        for i in range(self.num_ont):
            self.trafico_pendiente[i] +=  self.trafico_entrada[i] - self.trafico_salida[i]
            if self.trafico_entrada==0:
                pass
                #print(f"Cuando el trafico de entrada es 0 el pendiente es: {self.trafico_pendiente}")
            if self.trafico_pendiente[i] > 0:
                # Asegurarse de que el tráfico de salida en el siguiente ciclo considera el tráfico pendiente
                self.trafico_salida[i] = min(self.trafico_pendiente[i], self.Max_bits_ONT)
                self.trafico_pendiente[i] -= self.trafico_salida[i]
                # Asegurarse de que el tráfico pendiente no sea negativo
                #self.trafico_pendiente[i] = max(self.trafico_pendiente[i], 0)


        # Asegurarse de que la suma del tráfico de salida no supere la capacidad del OLT
        if np.sum(self.trafico_salida) > self.OLT_Capacity:
            #print("Entra aqui")
            exceso = np.sum(self.trafico_salida) - self.OLT_Capacity
            self.trafico_salida -= (exceso / self.num_ont)  # Distribuir el exceso entre todas las ONTs
            #self.trafico_salida = np.clip(self.trafico_salida, 0, self.Max_bits_ONT)

        # Calcular recompensa
        reward = self._calculate_reward()

        done = np.random.rand() > 0.99

        """
        elapsed_time = time.time() - start_time
        if elapsed_time < 0.002:
            time.sleep(0.002 - elapsed_time)
        """

        end_time = time.time()
        step_duration = end_time - start_time
        self.step_durations.append(step_duration)

        info = self._get_info()

        return self._get_obs(), reward, done, False, info


    def reset(self, seed=None, options=None):
        self.trafico_entrada, self.trafico_pareto_actual, self.trafico_pareto_futuro = self.calculate_pareto(self.num_ont, self.trafico_pareto_futuro)
        self.trafico_salida = np.random.uniform(low=self.Max_bits_ONT/10, high=self.Max_bits_ONT, size=self.num_ont).astype(np.float32)
        
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
