# DITTO: Offline Imitation Learning with World Models

[Link to paper](https://arxiv.org/abs/2302.03086)

## Getting Started
Code is currently considered pre-release, and not ready for significant external use. Beware.

To get started quickly, download the following tarball:
```sh
wget https://www.robots.ox.ac.uk/\~bdemoss/testdata.tar.gz
```

which includes 10 episodes from a strong PPO agent playing Breakout, as well as a converged world model for Breakout. The episodes should be put in their own directory, WM can be wherever.

Adjust the directories in src/config/test_config.yaml to match these directories (command line passing coming). Then, run main.py - then let the debugging begin.

## Other

Known numpy dependency bug, should be fixed with new gym envs from Farama coming Feb 23:

```
noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
```
is the secret problem

