# ADAPT: Adaptive Dispatch And Polling-free Targeting

Decentralised deep-RL load balancing for edge networks (CSD438 course project).
Every edge server runs a copy of one small shared DQN and decides every 0.25 s whether to
hold its buffered tasks or send them to itself or to one of up to five neighbours. The
baseline it is compared against is EVBLB (Sohrabi et al., GLOBECOM 2021).

Work in progress; see `DESIGN_NOTES.md` for every assumption we made on top of the brief.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
