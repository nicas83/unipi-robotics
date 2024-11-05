# unipi-robotics
## Robotics Projects: KANs vs MLPs: a comparison for a Robot Control Problem.

This work aims to answer to the question if a KANs [https://arxiv.org/abs/2404.19756] can improve accuracy on a static control problem for robot.
Starting from pykan library I built two different kind of models that learn direct and inverse static models on continuum robots by using KANs and compare the results with MLP on:
- Accuracy (based on Loss MSE)
- Computational complexity
- Continual Learning

More result on ProjectRoboticsKAN.pdf

For a correct compilation, download the pykan library from the following url https://github.com/KindXiaoming/pykan/tree/master
