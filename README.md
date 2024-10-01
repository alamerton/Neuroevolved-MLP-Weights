# Neuroevolution Algorithm for Artificial Neural Network Weight Optimization

## Introduction

This repository contains a genetic algorithm (GA) implementation for evolving the weights of an artificial neural network (ANN). The algorithm is designed to optimize the ANN's performance on three provided datasets. This project demonstrates how learning can be viewed as a search process and explores the effects of parameter changes on the ability of search techniques to solve optimization problems.

## Features

- Genetic Algorithm implementation for ANN weight optimization
- Customizable parameters for population size, number of generations, mutation rate, etc.
- Support for multiple datasets
- Performance visualization using matplotlib
- Comparison with other optimization algorithms (Cuckoo Search, Firefly Algorithm)

## Requirements

- Python 3.x
- matplotlib
- scikit-learn

## Usage

1. Clone the repository
2. Install the required dependencies
3. Run the main script: `python main.py`

## Parameter Settings

The algorithm uses the following parameter settings:

- Population Size: 50
- Number of Generations: 200
- Number of Iterations: 3
- Mutation Rate: 0.2
- Mutation Step: 0.8
- Crossover: False
- Gene Minimum: -4
- Gene Maximum: 4
- Train/Test Split: 67/33
- Number of Hidden Nodes: 8 (for the final dataset)

## Results

The algorithm demonstrates effective performance in optimizing ANN weights. The final parameter settings produce a mean lowest fitness of 25.93% and a lowest validation fitness of 27.0% on the largest dataset.

## Comparison with Other Algorithms

The report includes comparisons with other nature-inspired optimization algorithms:

- Cuckoo Search (CS)
- Firefly Algorithm (FA)

These comparisons provide insights into the strengths and weaknesses of different optimization approaches.

## Future Work

Potential future experiments and improvements include:

- Larger reflections of the population/generation ratio
- More parameters for mutation
- Increased number of iterations for more representative results
- Measuring the convergence of each algorithm setting
- Calculating mutation step using normal distribution
- Finding the Minimal Network for an Accurate Model
- Using the EA for hyperparameter setting of deep neural networks

## Contributors

- Alfie Lamerton

## License

## Contributing
Contributions to improve the algorithm or extend its functionality are welcome. Please feel free to submit pull requests or open issues for bugs and feature requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project was completed as part of a Biocomputation coursework assignment.
