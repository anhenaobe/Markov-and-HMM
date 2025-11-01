import re
import matplotlib.pyplot as plt
import numpy as np

'''
A Markov chain is a probabilistic model that represents a sequence of states, 
where the probability of transitioning to the next state depends only on a limited number 
of preceding states. Formally, this relationship can be expressed as:
P(Xn+1 | Xn=xn,Xn-1=xn-1,…,Xn-k+1=xn-k+1)
This means that to predict the next state X_n+1, 
it is sufficient to consider only the most recent k states of the process, rather than the 
entire history. Furthermore, Markov chains exhibit an important property known as the stationary 
state.This occurs when the probability distribution of the system no longer changes over time. 
Mathematically, this condition is expressed as:
πA = π
where A represents the transition matrix, which contains the probabilities of moving from one 
state to another, and π denotes the stationary distribution vector. In this state, the system 
reaches equilibrium, meaning that the likelihood of being in each state remains constant across 
transitions.
'''

class Markov:
    def __init__(self, text, words=False, tokens=1):
        self.text = text
        self.words = words
        self.tokens = tokens

    @staticmethod
    def clean(text):
        clean_text = text.lower()
        clean_text = re.sub(r'[^\w\s]', '', clean_text)
        return clean_text

    @staticmethod
    def split(text):
        return text.split()

    @staticmethod
    def index_matrix(states):
        n = len(states)
        index = {state: i for i, state in enumerate(states)}
        return np.zeros((n, n), dtype=float), index
        
    def transition_matrix(self):
        t = self.tokens
        words = self.words

        cleaned = Markov.clean(self.text)
        cleaned = Markov.split(cleaned) if words else list(cleaned)

        if len(cleaned) == 0:
            return np.array([]), [], {} 
        
        if t != 1:
            transitions = {}
            states = []
            for i in range(len(cleaned) - t):
                group = tuple(cleaned[i : i + t])
                next_t = cleaned[i + t]
                if group not in transitions:
                    transitions[group] = []
                transitions[group].append(next_t)
                states.append(group)
            states = sorted(set(states))
            matrix, index = Markov.index_matrix(states)
            for group in transitions:
                for next_t in transitions[group]:
                    next_group = tuple(list(group[1:]) + [next_t])
                    if next_group in index:
                        matrix[index[group], index[next_group]] += 1
                
        else:
            states = sorted(set(cleaned))
            matrix, index = Markov.index_matrix(states)
            for i in range(len(cleaned) - 1):
                current = cleaned[i]
                next_ = cleaned[i + 1]
                matrix[index[current], index[next_]] += 1

        row_sums = matrix.sum(axis=1, keepdims=True)
        zero_rows = (row_sums.squeeze() == 0)
        row_sums[row_sums == 0] = 1.0
        matrix = matrix / row_sums

        # filas que quedaron todas ceros → asignar distribución uniforme
        if zero_rows.any():
            matrix[zero_rows, :] = 1.0 / matrix.shape[1]

        return matrix, states, index
        

    def text_generator(self, length=200):
        matrix, states, index = self.transition_matrix()
        t = self.tokens
        if matrix.size == 0:
            return ""
        n = matrix.shape[0]
        prob = np.ones(n) / n
        prob /= prob.sum()
        current_state = np.random.choice(n, p=prob)
        
        if t == 1:
            generated = [states[current_state]]
            for _ in range(length - 1):
                p = matrix[current_state]
                if not p.any():
                    p = np.ones(n)/n
                p /= p.sum()
                current_state = np.random.choice(n, p=p)
                generated.append(states[current_state])
            return ''.join(generated)

        else:
            generated = list(states[current_state])   
            for _ in range(length - t):
                p = matrix[current_state]
                if not p.any():
                    p = np.ones(n) / n
                p /= p.sum()
                next_index = np.random.choice(n, p=p)
                next_state = states[next_index]      
                new_token = next_state[-1]            
                generated.append(new_token)
                current_state = next_index  
            return ' '.join(generated)
    
    def stationary_distribution(self):
        matrix, states, index = self.transition_matrix()
        vals, vecs = np.linalg.eig(matrix.T)
        idx = np.argmin(np.abs(vals - 1))
        vec = np.real(vecs[:, idx])
        pi = vec / vec.sum()
        pi = np.real_if_close(pi)
        pi[pi < 0] = 0
        return pi / pi.sum()
    
    def simulate_chain(self, steps=1000):
        matrix, states, index = self.transition_matrix()
        if matrix.size == 0:
            return np.array([])
        n = matrix.shape[0]
        current = np.random.choice(n)
        counts = np.zeros(n)
        for _ in range(steps):
            counts[current] += 1
            current = np.random.choice(n, p=matrix[current])
        freqs = counts / steps
        return freqs


def main():
    with open('text.txt', 'r', encoding='utf-8') as f:
        text = f.read()  

    markov = Markov(text, True, 5)
    generated = markov.text_generator(60)
    print(generated)
    matrix, states, index = markov.transition_matrix()
    with open('output.txt', 'w', encoding='utf-8') as o:
        o.write(generated)

    print(markov.simulate_chain())


#to plot this seccion, it is necesary to have enough computational power
'''
def compare_stationary_vs_empirical(matrix, states, text):
    markov = Markov(text, True, 4)
    pi = markov.stationary_distribution()
    freqs = markov.simulate_chain()
    print("Distribución estacionaria (π):")
    print(dict(zip(states, np.round(pi, 4))))
    print("\nFrecuencias empíricas (10 000 pasos):")
    print(dict(zip(states, np.round(freqs, 4))))

    steps = 2000
    n = matrix.shape[0]
    probs = np.zeros((steps, n))
    current = np.random.choice(n)
    counts = np.zeros(n)
    for i in range(steps):
        counts[current] += 1
        probs[i] = counts / (i + 1)
        current = np.random.choice(n, p=matrix[current])

    plt.plot(probs)
    plt.axhline(y=pi.mean(), color='k', linestyle='--', label='Media de π')
    plt.title("Convergencia de frecuencias hacia la distribución estacionaria")
    plt.xlabel("Paso")      # corregir aquí
    plt.show()
'''
if __name__ == '__main__':
    main()