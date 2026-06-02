# Problem Definition

What problem does this paper address? Write this so that someone with no knowledge of the paper can understand what needs to be solved and how success is measured.

Do NOT describe the paper's method or solution here — only the problem itself. Think about if a future scientist agent try to solve this problem from scratch without knowing what approach the paper took. If you leak the solution, you compromise the benchmark.

## Research Question

Estimate a user's 3-D arm pose from sensors available on common mobile devices, without cameras, external infrastructure, or per-user training data. The target output is shoulder, elbow, and wrist joint position over time in a body-centered coordinate frame.

## Why It Matters

Accurate mobile arm tracking would support fitness, rehabilitation, life logging, occupational training, games, and context-aware interfaces while avoiding privacy and deployment drawbacks of camera-based or instrumented motion-capture systems.

## Success Criteria

A successful system predicts wrist and elbow positions with low centimeter-scale error on held-out users, remains stable across natural activities and large arm motions, and does not require user-specific training examples.
