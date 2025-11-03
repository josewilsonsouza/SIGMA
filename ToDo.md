# Modelos
1. `sigma.py`: usa o ponto D para formulação do score de convexidade.
* Regra de atualização:
$$\theta_{t+1} = \theta_{t} - \eta\cdot g_t\odot \sigma_t$$
* Momentum
* Media móvel exponencial
2. `sigma_v2.py`: Faz um mix do adam com o SIGMA usando o ponto D
3. `sigma_v2.py`: Faz um mix do adam com o SIGMA usando o ponto C
4. 