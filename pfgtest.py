#%%
from pfg.factor_graph import FactorGraph, Variable, Factor, FactorCategory
import numpy as np
import pprint

fg = FactorGraph()
# %%
var_a = Variable('Alice', 5)
var_b = Variable('Bob', 5)
var_c = Variable('Carol', 5)
# %%
factor_apt_a = Factor(np.array([0.05, 0.05, 0.3, 0.3, 0.3]), name='Aptitude_Alice')
factor_apt_b = Factor(np.array([0.2, 0.3, 0.3, 0.2, 0.0]), name='Aptitude_Bob')
factor_apt_c = Factor(np.array([0.2, 0.2, 0.2, 0.2, 0.2]), name='Aptitude_Carol')
fg.add_factor([var_a], factor_apt_a)
fg.add_factor([var_b], factor_apt_b)
fg.add_factor([var_c], factor_apt_c)
# %%

def correlation_value(a, b, c):
    return 1 - 0.1 * (abs(a - b) + abs(b - c) + abs(a - c))

corr_values = np.zeros([5, 5, 5])

for a in range(5):
    for b in range(5):
        for c in range(5):
            corr_values[a, b, c] = correlation_value(a, b, c)
          
print('Correlation Tensor:')
print(corr_values)
            
# ----------

corr_factor = Factor(corr_values, name='Correlation')

fg.add_factor([var_a, var_b, var_c], corr_factor)
