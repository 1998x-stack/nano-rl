# nano-rl V2 · Architecture and Verification Design

## Thesis
The original curriculum covers value learning → policy gradient → deep RL → offline/group updates. V2 completes five distinct branches **without claiming paper-scale reproduction**:

| Stage | True mechanism implemented | Deliberately simplified |
|---|---|---|
| 11 AlphaZero | symmetric 2-player zero-sum TicTacToe, neural policy/value, PUCT search, visit targets, self-play | no distributed replay / chess |
| 12 MuZero-inspired | h representation, g learned latent transition + reward, f value/policy, unroll training, latent PUCT search | supervised dynamics/value teacher, no reanalyse |
| 13 Dreamer-inspired | posterior/prior recurrent stochastic state-space model, reconstruction + KL, learned reward, imagined actor-critic | basic diagonal Gaussian RSSM; no DreamerV3 symlog/discretization |
| 14 Decision Transformer | return-conditioned causal token interleaving and masked action prediction, fixed behavior trajectories | small MLP-embed transformer over finite states, not Atari |
| 15 RLHF | supervised starting policy, pairwise Bradley-Terry reward model, frozen reference, PPO-style preference optimization | one-token categorical responses, synthetic preferences instead of human labels |

These are *distinct* approaches and not a chronological linear chain.

## Data lineage & no-leak design
- 11 generates games with true rules only in self-play. MCTS uses legal moves and terminal rules (as AlphaZero does); labels use final game outcome from the acting player's perspective.
- 12 collects controlled corridor transitions **offline** to supervise the latent model. Planning thereafter uses only learned dynamics g, not the corridor oracle; test searches and environment evaluation are separated.
- 13 trains a world model from collected real transitions, then uses **imagined latent rollouts** for actor/value improvement.
- 14 uses fixed offline behavior episodes. Every training action logit is read from the *state token position*, before the same-step action token, using a triangular causal attention mask. The desired return is available as an inference-time control.
- 15 preference labels are algorithmically generated from exact mathematical correctness. They stand in for a preference annotator; the reward model must not see true labels during PPO.

## Interface
For every new chapter: `train.py --help`, `--steps` or `--games`, `--seed`, `--no-log`, chapter README, standalone HTML, paper provenance, tests for core invariants. Shared `utils/` exposes only environment, reproducibility, logging. Losses/search remain visible in chapters.

## Definition of Done
1. All stage code is readable and standalone under 500 lines per chapter.
2. New mathematical and regression tests pass: perspective flip and legality, value backprop, learned-vs-oracle boundary, KL/reconstruction finite, token causality, Bradley-Terry ranking and PPO ratio.
3. CPU smoke tests run all 15 stages in GitHub Actions; test duration is small.
4. HTML pages show **computed quantities**, not fabricated training success curves, and validate inline JS syntax.
5. Data origins, differences from published algorithms, and licensing are explicit.
6. Pages workflow does not fail when Pages is not enabled; initial activation stays with repo owner.

## Evaluation protocol
Quick smoke tests verify shape, finite loss, no leakage, no exceptions, not convergence. Real training uses fixed seeds 0,1,2,3,4; logs episode return / policy accuracy / model consistency as appropriate; report mean±standard error, environment steps, wall time, and budget. Never compare algorithms from different observation/interaction budgets as if fair.

## Design trade-offs
**Faithful kernel over superficial breadth**: The hard-to-implement components (large self-play infrastructure, observation support projections, LLM reward models) are not faked. Explicit extension points in each README guide advanced reproductions; a toy pipeline with 100% core invariants is preferable to copying a research framework.

**Risk register**: Model bias in latent search, reward model misspecification, distribution shift, MCTS leaf-value sign bugs, offline action leakage, KL collapse, asynchronous on-policy staleness. Tests and visualizations target these failure modes.
