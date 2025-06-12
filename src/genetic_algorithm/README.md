
# Genetic Algorithm for Antenna Placement Optimization

## Overview
This document describes the implementation of a Genetic Algorithm (GA) used to optimize the placement of antennas in a grid-based cellular network.  
The goal is to **maximize network performance** by minimizing user connection failures and service redirects over time.

---

## Function: `optimize_with_genetic_algorithm`

The function takes the following parameters:

| Parameter        | Type   | Default | Description                                                  |
|------------------|--------|---------|--------------------------------------------------------------|
| `env`            | Object | —       | The environment simulating the cellular network.             |
| `population_size`| int    | 20      | Number of individuals in each generation.                   |
| `generations`    | int    | 50      | Number of evolutionary cycles.                               |
| `mutation_rate`  | float  | 0.1     | Probability of mutation for each individual.                 |
| `simulation_steps` | int  | 8670    | Number of steps for fitness evaluation (e.g., 8670 = 1 year).|

---

## GA Workflow Summary

1. **Initialization**: Individuals are created with a random number of antennas between computed minimum and maximum values.  
2. **Evaluation**: Each individual is simulated for `simulation_steps` to compute its average reward (fitness).  
3. **Selection**: Tournament selection is used to choose better individuals for reproduction.  
4. **Crossover**: Single-point crossover combines parents into children.  
5. **Mutation**: Children undergo mutation (add, remove, or move antennas) with a given mutation rate.  
6. **Elitism**: The best individual from the previous generation is preserved.  
7. **Repeat**: The process continues for the specified number of generations.
