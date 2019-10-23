num_states = 4
num_actions = 2

q = {s: {a: 0.0 for a in range(num_actions)} for s in range(num_states)}

print(list(q[0].values()))