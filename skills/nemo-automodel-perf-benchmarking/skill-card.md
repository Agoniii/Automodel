## Description: <br>
Measure and attribute training-throughput changes in NeMo AutoModel on a shared Slurm GPU cluster: paired A/B/A2 jobs, step-time and tok/s cross-checks, numerics gates, nsys window analysis, and the evidence a performance PR needs. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Engineers evaluating kernel, configuration, or parallelism changes for throughput and memory on multi-node GPU clusters, and reviewers deciding whether a reported gain is real. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Benchmark-only settings (force-balanced MoE gate, synthetic data) could be copied into training configurations. <br>
Mitigation: The skill labels benchmark-only settings and requires them to be tagged in results; review configurations before training. <br>
Risk: Submitting memory-exceeding configurations can destabilize shared nodes. <br>
Mitigation: The skill requires memory estimates and owner approval before every GPU job. <br>

## Reference(s): <br>
- [NeMo AutoModel Documentation](https://docs.nvidia.com/nemo/automodel/latest/index.html) <br>
- [NeMo AutoModel GitHub Repository](https://github.com/NVIDIA-NeMo/Automodel) <br>

## Skill Output: <br>
**Output Type(s):** [Measurement plans, Slurm job sketches, result tables, PR evidence sections] <br>
**Output Format:** [Markdown with inline bash and Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- `claude-code` <br>

## Evaluation Tasks: <br>
Three evaluation tasks covering paired-measurement design, interpretation of a noisy A/B result, and host-bound versus GPU-bound diagnosis from an nsys timeline. <br>
