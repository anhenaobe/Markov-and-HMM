import numpy as np
from hmmlearn import hmm
from sklearn.feature_extraction.text import CountVectorizer

# 1. Dataset básico 
texts = [
    "I love this movie, it was amazing",
    "This is the best day ever",
    "I hate this, it was terrible",
    "The experience was awful and bad",
    "It was okay, not great but not bad either"
]

# 2. Asignamos etiquetas manuales (para entrenar)
# 0 = negativo, 1 = neutro, 2 = positivo
labels = [2, 2, 0, 0, 1]

# 3. Convertimos texto a vectores (bolsa de palabras)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()

# 4. Configuramos el modelo HMM (Gaussian)
model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)

# 5. Entrenamiento
model.fit(X)

# 6. Predicción de estados ocultos
hidden_states = model.predict(X)
print("Estados ocultos predichos:", hidden_states)

# 7. Asociación con etiquetas conocidas (para interpretación)
state_labels = {
    0: "Negativo",
    1: "Neutro",
    2: "Positivo"
}

for i, text in enumerate(texts):
    print(f"Texto: {text}")
    print(f"→ Estado inferido: {state_labels[hidden_states[i]]}")
    print("------")
