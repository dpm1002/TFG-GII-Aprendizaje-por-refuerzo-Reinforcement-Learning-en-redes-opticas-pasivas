import numpy as np

import gymnasium as gym
from gymnasium import spaces

import time

from scipy.stats import pareto
from scipy.stats import beta as beta_dist

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

        #Velocidad maxima de la red
        self.Vt=Vt

        #Tiempo de cada ciclo: 2ms
        self.temp_ciclo=0.002

        #Capacidad maxima del OLT
        self.OLT_Capacity=Vt*self.temp_ciclo

        #El espacio de observacion sera el de la capacidad de cada ONT, definido con un array
        #del valor de cada una de las capacidades
        self.observation_space=spaces.Box(low=0, high=self.OLT_Capacity, shape=(self.num_ont,), dtype=np.float32)

        #Definimos el espacio de acciones, en nuestro caso es el numero de ONTs sobre las 
        #que trabajaremos
        self.action_space = spaces.Box(low=-self.OLT_Capacity / 20, high=self.OLT_Capacity / 20, shape=(self.num_ont,), dtype=np.float32)

        #Registro de los estados de ON y OFF
        self.on_off_state=False

        # Registros de tiempo
        self.step_durations = []

        self.trafico_entrada=[]

        self.trafico_pareto_futuro=[]

        self.trafico_salida=[]

        self.trafico_pareto_actual=[]
        
        self.last_reward=0

        self.tamano_cola=0

        # Inicialización del estado
        self.state = None
        self.reset()

    def _get_obs(self):
        # Asegurar que el estado siempre está dentro de los límites definidos por el espacio de observaciones
        obs = np.clip(self.trafico_entrada, 0, self.OLT_Capacity)
        return np.squeeze(obs)
    
    def _get_info(self):

        info={
            'OLT_Capacity':self.OLT_Capacity,
            'trafico_entrada':self.trafico_entrada,
            'trafico_salida':self.trafico_salida,
            'trafico_IN_ON_actual':self.trafico_pareto_actual,
            'trafico_IN_ON_futuro':self.trafico_pareto_futuro,
            'tamano_cola':self.tamano_cola
        }

        return info

    def calculate_pareto(self, num_ont=5, traf_pas=[]):
        # Parámetros de la distribución de Pareto
        alpha_ON = 2  # Parámetro de forma (alfa)
        alpha_OFF = 1  # Parámetro de forma (alfa)
        #Vel_tx_max = 2e9 # 2 Gbps de velocidad de transmisión máxima
        Vel_tx_max=self.Vt

        # Guardamos los valores de los bits de cada ont en esta lista para la cola de la ont
        trafico_futuro_valores=[]

        lista_trafico_act=[]

        trafico_actual_lista=[[] for i in range(self.num_ont)]

        for i in range(num_ont):
            #print(i)
            # Generar valores aleatorios de la distribución de Pareto para cada ont
            # Para cada ont, generamos 'num_samples' valores
            #De momento solo creo una ont
            if(traf_pas == []):
                trafico_pareto = list(np.random.pareto(alpha_ON, size=(1)))
                trafico_pareto = trafico_pareto+(list(np.random.pareto(alpha_OFF, size=(1))))
                trafico_pareto = list(trafico_pareto)
            else:
                trafico_pareto = traf_pas[i]

            #print(trafico_pareto)

            suma=sum(trafico_pareto)
            #print(suma)
            #Debe de ser menor a 2 milisegundos
            while(suma<2):
                trafico_pareto=trafico_pareto+(list(np.random.pareto(alpha_ON, size=(1))))+(list(np.random.pareto(alpha_OFF, size=(1))))
                #print(trafico_pareto)
                suma=sum(trafico_pareto)
                #print(suma)

            traf_act=[]
            suma=0
            while(suma<2):
                traf_act.append(trafico_pareto.pop(0))
                suma=sum(traf_act)

                #print(traf_act)
                #print(suma)  
            traf_fut=[0, 0]
            if(len(traf_act)%2==0):
                traf_fut[0]=0
                traf_fut[1]=suma-2
                traf_act[-1]=traf_act[-1]-traf_fut[1]
            else:
                traf_fut[0]=suma-2
                traf_fut[1]=trafico_pareto[-1]              
                traf_act[-1]=traf_act[-1]-traf_fut[0]

            trafico_actual_lista[i].append(traf_act)    
                #print(traf_fut)
            vol_traf_act=sum(traf_act[::2])*Vel_tx_max*self.temp_ciclo

            lista_trafico_act.append(vol_traf_act)
            trafico_futuro_valores.append(traf_fut)
            #print(vol_traf_act)
            #print(traf_act)

        return lista_trafico_act, trafico_actual_lista, trafico_futuro_valores

    def _calculate_reward(self):
        
        total_desviacion = np.sum(np.abs(self.trafico_entrada - self.trafico_salida))
        reward = -total_desviacion
        return reward

    def step(self, action):
        start_time = time.time()

        aux=False

        while aux==False:

            # Obtener el tráfico de entrada actual
            self.trafico_entrada, self.trafico_pareto_actual, self.trafico_pareto_futuro = self.calculate_pareto(self.num_ont, self.trafico_pareto_futuro)

            # Aplicar la acción para ajustar el tráfico de salida hacia el tráfico de entrada
            self.trafico_salida = np.clip(self.trafico_entrada + action, 0, self.OLT_Capacity)

            # Cálculo del tamaño de la cola
            aux_tamano_cola = np.sum(self.trafico_entrada - self.trafico_salida)

            if aux_tamano_cola<=self.OLT_Capacity:
                aux=True
        
        
        if aux_tamano_cola<0:
            self.tamano_cola=0
        else:
            self.tamano_cola=aux_tamano_cola
        # Calcular recompensa
        reward = self._calculate_reward()

        # Ajustes basados en la recompensa
        if self.last_reward is not None and reward < self.last_reward:
            adjustment = np.random.uniform(0, 20)
            self.trafico_salida = np.where(self.trafico_entrada == 0, 0, self.trafico_salida)
            self.trafico_salida = np.where(self.trafico_entrada > self.trafico_salida, self.trafico_salida + adjustment, self.trafico_salida - adjustment)
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
        

    def reset(self,seed=None, options=None):

        self.trafico_entrada, self.trafico_pareto_actual, self.trafico_pareto_futuro=self.calculate_pareto(self.num_ont, self.trafico_pareto_futuro)

        self.trafico_salida = np.random.uniform(low=0, high=self.OLT_Capacity, size=self.num_ont).astype(np.float32)
        
        #Definimos los valores de observacion y de info
        observation = self._get_obs()
        info = self._get_info()
        return observation, info 


    def render(self, mode='human', close=False):
        if mode == 'human':
            #print(f"Estado actual del tráfico: {self.state}")
            pass

from gymnasium.envs.registration import register

register(
    id="RedesOpticasEnv-v0",
    entry_point="custom_env.redes_opticas_env:RedesOpticasEnv",
    max_episode_steps=300,
)