# SmartPoser: Arm Pose Estimation with a Smartphone and Smartwatch Using UWB and IMU Data

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2509.03451"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2509.03451-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2509.03451](https://arxiv.org/abs/2509.03451) |
| **Domain** | Mobile sensing / wearable pose estimation |
| **TL;DR** | SmartPoser fuses phone-watch UWB range with IMU streams to estimate 3-D arm pose using commodity devices. |

## Summary

The paper targets mobile arm pose estimation without cameras, external beacons, or extra worn sensors. It proposes a two-stage recurrent model: one network corrects noisy UWB range and another predicts shoulder, elbow, and wrist positions from corrected range plus IMU features. The original study reports centimeter-scale tracking on held-out users and shows that adding UWB improves over IMU-only sensing.

## Problem

Estimate a user's 3-D arm pose from sensors available on common mobile devices, without cameras, external infrastructure, or per-user training data. The target output is shoulder, elbow, and wrist joint position over time in a body-centered coordinate frame.

## The paper's method

SmartPoser estimates arm pose using only an off-the-shelf smartphone and smartwatch by combining inertial measurements with phone-watch UWB distance. The central claim is that absolute UWB range stabilizes and improves arm tracking compared with IMU-only sensing.

## Key results of the paper

- Median elbow+wrist error: 11.0 cm in the combined study data.
- UWB correction reduces mean range error from 21.0 cm to 8.6 cm.
- IMU-only ablation worsens median elbow+wrist error from 11.0 cm to 14.2 cm.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.5-xhigh` | +165.7 |
| `gpt-5.6-terra-xhigh` | +110.2 |
| `gpt-5.6-sol-xhigh` | +99.0 |
| `gpt-5.6-luna-xhigh` | +25.0 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2509.03451 --mode innovate --agent claude-code
sw evaluate 2509.03451
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

