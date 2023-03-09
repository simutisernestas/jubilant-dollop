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

Execute `./ckf_sim.py` to run simulation.
