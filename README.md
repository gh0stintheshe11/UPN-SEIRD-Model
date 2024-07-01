# Modelling Impact of Positive and Negative Information on SEIRD-like Epidemics in Multiplex Networks

This repository contains code and resources for studying the influence of positive and negative information dissemination on SEIRD-like epidemics within multiplex networks using the UPN-SEIRD model. The model is validated through Barabási–Albert (BA) and Monte Carlo (MMC) simulations.

## Introduction

Epidemic spreading models are crucial for understanding infectious disease dynamics, especially considering the complex nature of real-world interactions. Traditional models like SEIRD provide insights but often neglect the impact of information dissemination. This study addresses this gap by investigating the coevolution of information and disease spread in multiplex networks using the UPN-SEIRD model.

### UPN-SEIRD Model Overview

The UPN-SEIRD model captures the interaction between disease spread and information dissemination across two interconnected layers:
- **Physical Contact Layer:** Represents face-to-face interactions where the disease spreads.
- **Virtual Communication Layer:** Simulates the spread of information through online or mass media channels.

## Usage
To install the required dependencies, run:

```bash
pip install -r requirement.txt
```

### UPN & SEIRD standalone simulation
  * infomation layer -> BA
  * epidemic layer -> GIRG
    
### UPN-SEIRD hybrid model simulation
  * both layers -> BA


```python
# Example usage:
n = 1000  # number of nodes
d = 2  # dimension
tau = 2.5  # power-law exponent
alpha = 2  # distance dependence exponent
expected_weight = 1  # expected weight

G = GIRG(n, d, tau, alpha, expected_weight).graph

# Visualize the network
visualize_network(G)
```

## Results

The simulations and visualizations provide insights into how information dissemination influences epidemic dynamics in multiplex networks.





