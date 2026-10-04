"""MechaVerify: a procedurally generated, programmatically verifiable environment
for grounded mechanism and manufacturability reasoning in vision-language models.

Public surface:
    - mechaverify.specs.Instance / Answer  : the data model for a benchmark item
    - mechaverify.rng.SeededRNG            : deterministic, reproducible sampling
    - mechaverify.verify                   : deterministic answer parser + checker
    - mechaverify.tasks                    : task-family generators (kinematics, DFMA)
    - mechaverify.render                   : engineering-drawing-style renderers
    - mechaverify.dataset                  : dataset assembly + manifest I/O
"""

__version__ = "0.1.0"

from mechaverify.specs import Answer, Instance  # noqa: F401
