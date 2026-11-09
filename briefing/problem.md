# Problem Definition

## Research Question

Estimate a user's 3-D arm pose from sensors available on common mobile devices, without cameras, external infrastructure, or per-user training data. The target output is shoulder, elbow, and wrist joint position over time in a body-centered coordinate frame.

## Why It Matters

Accurate mobile arm tracking would support fitness, rehabilitation, life logging, occupational training, games, and context-aware interfaces while avoiding privacy and deployment drawbacks of camera-based or instrumented motion-capture systems.

## Success Criteria

A successful system predicts wrist and elbow positions with low centimeter-scale error on held-out users, remains stable across natural activities and large arm motions, and does not require user-specific training examples.

## Benchmark Setting

The paper's recordings are not available, so this benchmark uses a simulated substitute with the same sensing setup: a phone in the left front pocket and a watch on the left wrist. Phone-watch range arrives at 5 Hz, is held to 25 Hz, and carries body-occlusion bias and noise. The phone and watch orientation estimates drift. Accelerations are expressed in the phone frame. Arm-span-normalized shoulder/elbow/wrist positions serve as ground truth. Ten development participants (8 minutes each: everyday activity, then wide gross-motor movement) are visible. Systems are scored on 48 unseen participants from the same distribution. See `briefing/evaluation.md` for the submission interface and the measurement.
