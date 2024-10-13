# FastSLAM

Courtesy of https://github.com/yingkunwu/FastSLAM


### Introduction

The purpose of this project is to simulate a Simultaneously Localization and Mapping problem by allowing a robot with beam sensors to navigate in a 2D grid map. In this project, I implemented both the famous particle filter based SLAM algorithm, FastSLAM1.0 and FastSLAM2.0, in python. The program is able to work in any given grid map as long as the environment is static and each grid is either free (0) or occupied (1).

### Usage

Run the following command to run FastSLAM1.0:
```shell
python fastslam1.py -m [given map] -p [number of particles]
```

Similarly, run the following command to run FastSLAM2.0:
```shell
python fastslam2.py -m [given map] -p [number of particles]
```

For now the given map should be either 'scene-1' or 'scene-2', and the number of particles should be a integer.

### Result

#### FastSLAM 1.0

<img src="https://github.com/kunnnnethan/FastSLAM/blob/main/result/fastslam1-scene1.png" alt="result" width=70% height=70%/>

<img src="https://github.com/kunnnnethan/FastSLAM/blob/main/result/fastslam1-scene2.png" alt="result" width=70% height=70%/>

#### FastSLAM 2.0

<img src="https://github.com/kunnnnethan/FastSLAM/blob/main/result/fastslam2-scene1.png" alt="result" width=70% height=70%/>

<img src="https://github.com/kunnnnethan/FastSLAM/blob/main/result/fastslam2-scene2.png" alt="result" width=70% height=70%/>


### References

* Michael Montemerlo, Sebastian Thrun, Daphne Koller, and Ben Wegbreit. Fastslam: A factored solution to the simultaneous localization and mapping problem.
* Giorgio Grisetti, Cyrill Stachniss, and Wolfram Burgard. Improved techniques for grid mapping with rao- blackwellized particle filters.
* https://github.com/udacity/RoboND-MCL-Lab
* https://github.com/ivanwong9290/CMU_16833_SLAM/tree/ae5a127fadb64b9cf3e2de51b26d2ad339e50b6a
* https://bostoncleek.github.io/project/fast-slam
* https://github.com/p16i/particle-filter/tree/5d8a82ea95b93b2d868435a797e5c8c6bdc5de56

### Dependencies

You need to install `opencv-python` and `matplotlib`. After installing `matplotlib` you might need to downgrade your `numpy` to version 1.23.

### What you need to do

- Note that this implementation uses a simple model for the _robot_. In your case you need to use a model that reflects the motion of the car. Note that you do not need to drive fast when doing the SLAM, keep it slow. Then using kinematic bicycle model would be valid.
- In the current implementation the control inputs are pre-determined and are loaded using `config.yaml`. In your case you need to calculate them on-the-fly.
- Use a simple controller to travel along the circuit (e.g. your wall following algorithm) to collect measurements and complete the localisation and mapping task.