"""In-memory stand-ins for weights, accelerators, and agents.

They satisfy the protocols in ``honing.core`` with ordinary Python, so the
walking skeleton can run a whole campaign on a CPU in milliseconds. Each fake
models only the behavior a test needs to observe: training raises quality,
cutting capacity lowers it, quantizing costs a little unless the weights were
trained for the scheme, and a verification instruction helps an agent.
"""
