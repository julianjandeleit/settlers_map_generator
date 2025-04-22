#%%
import numpy as np
import factorgraph as fg

# Make an empty graph
g = fg.Graph()

# Add some discrete random variables (RVs)
g.rv('a', 5)

# Add some factors, unary and binary
g.factor(['a'], potential=np.array([0.1, 0.1,0.2,0.3,0.3]))
g.factor(['a'], potential=np.array([
0.1,0.1,0.2,0.3,0.3
]))

# Run (loopy) belief propagation (LBP)
iters, converged = g.lbp(normalize=True)
print('LBP ran for %d iterations. Converged = %r' % (iters, converged))
print()

# Print out the final messages from LBP
g.print_messages()
print()

# Print out the final marginals
g.print_rv_marginals(normalize=True)
# %%
rvs = g.get_rvs()
g.rv_marginals([rvs["a"]],normalize=True)
# %%
