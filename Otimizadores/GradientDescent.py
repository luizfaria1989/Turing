class GradientDescent:

    """
    Implementa o método do gradiente com a fórmula x' = x - \nabla f(x)
    Parâmetros:
    learningRate: Representa a taxa de aprendizado, o tamanho do passo que o modelo irá deslocar
    de um ponto a outro
    function: Representa a função que se quer descobrir o ponto de mínimo / otimizar
    functionPrime: Representa a derivada da função que se quer otimizar
    initialPoint: Representa o ponto inicial que o algoritmo irá começar o procurar pelo ponto de mínimo
    maxIterations: Representa a quantidade máxima de iterações que o algoritmo irá aplicar o método
    do gradiente
    tolerance: Representa a distância que a norma do gradiente estará de zero
    Retornos:
    Retorna o ponto de mínimo da função ou o último ponto em que parou até a última iteração dada por
    maxIterations
    path: É uma lista que armazena os pontos que o modelo dá em direção ao ponto de mínimo da função
    """

    def __init__(self, function, function_prime, initial_point, learning_rate=0.001, max_iterations=100, tolerance=1e-6):
        self.f = function
        self.fp = function_prime
        self.ip = initial_point
        self.lr = learning_rate
        self.iterations = max_iterations
        self.tol = tolerance
        self.path = []

    def update_step(self):
        for i in range(self.iterations): # Inicia o Loop do Método do Gradiente
            self.path.append(self.ip) # Adiciona o novo ponto na lista de pontos (path)
            grad = self.fp(self.ip) # Calcula o valor do vetor gradiente no ponto
            if abs(grad) < self.tol: break # se o gradiente for 0 então convergiu
            self.ip = self.ip + self.lr * (-grad) # Atualiza o valor do ponto
        return self.ip, self.path
