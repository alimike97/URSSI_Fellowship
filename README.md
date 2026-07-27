# Silent Failures in ML-Driven Simulations

This repository is part of US Research Software Sustainability Institute (URSSI) Summer 2026 Early Career Fellowship. It aims to produce an open-source diagnostic framework for detecting subtle reliability failures in scientific simulations that use machine learning (ML), reduced-order models (ROMs), neural operators, and other learned components.

Scientific ML models can produce results that look numerically plausible even after the simulation has entered a physically invalid or unreliable regime. This project aims to develop practical tools to identify those failures during runtime, before they contaminate scientific conclusions.

## Project goals

This project will deliver four connected resources:

1. **Failure-mode taxonomy and database** - a structured classification of silent failures, their physical characteristics, detection signatures, likely causes, and scientific consequences.
2. **Runtime diagnostic library** - extensible monitors and interfaces for detecting reliability problems while an ML-driven simulation is running.
3. **Benchmark suite** - reproducible, parameterized simulation cases designed to trigger representative failure modes.
4. **Educational materials** - a module showing researchers how to add runtime reliability checks to scientific ML workflows.

The initial testbed uses computationally affordable one-dimensional fluid problems, including compressible-flow, shock-dominated, detonation, and combustion cases. The interfaces and concepts are intended to be extensible to other scientific simulation domains.

The initial fluid-dynamics benchmarks build on [CompFlowLab](https://github.com/alimike97/CompFlowLab), an open-source Python environment for prototyping data-driven models for compressible flows with shocks and chemical reactions. CompFlowLab provides an efficient foundation for reproducing controlled failure scenarios, while this project focuses on failure classification, runtime diagnosis, and reliability tooling.

## Project timeline
See [TIMELINE.md](TIMELINE.md) for milestones, deliverables, and completion criteria.

## Acknowledgment
This work is supported by the US Research Software Sustainability Institute (URSSI) via grant G-2022-19347 from the Sloan Foundation.



