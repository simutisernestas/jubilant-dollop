## Collaborative EKF Localization 

Filter is implemented on UWB range measurements & IMU. The code is based mostly on these papers (with some trivial changes):

1. https://journals.sagepub.com/doi/abs/10.1177/0278364918760698
2. https://arxiv.org/abs/2104.14106

## Install

Project depends on: https://github.com/Aceinna/gnss-ins-sim for generating simulated IMU data. Install:

```bash
git clone https://github.com/Aceinna/gnss-ins-sim
cd gnss-ins-sim
pip install -e .
```

Other dependencies:

```
pip install -r requirements.txt
```

## Data

Run `./gen_data.py` to generate data used for experiments.

## Run

Execute `./ckf_sim.py` to run simulation. Main implementation is in `ckf.py` class.

## Docker

```
docker build -t myapp .
docker run -it --rm -v "$(pwd)":/app -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix myapp
```
