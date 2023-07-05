Simple example of implementing dynamic binding (with optional induced charge).

Description of the parameters in the input script
Output file (bonds_out) describes the connectivity within the system.

Sample from the output file:

# TIMESTEP, no. of bonds, no. of free bonds, no. of total possible bonds
# List of bond partners (donor particle id - acceptor particle id)

```
TIMESTEP: 800 2 3 5
8 7
10 9
```

Runs with the command: ../../matilda.ft -particle