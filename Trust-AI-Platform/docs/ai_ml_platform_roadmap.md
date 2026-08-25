<div class="cover-page">
  <div class="cover-mark">PLATFORM DEVELOPMENT ROADMAP</div>
  <img class="cover-logo" src="../static/img/logo.png" alt="Trust AI Lab logo">
  <h1>AI, Machine Learning & Platform Improvements</h1>
  <p class="cover-subtitle">A practical roadmap for trustworthy analytics, adaptive learning, teacher support, and measurable educational impact</p>
  <div class="cover-flow">
    <span>CLEAN DATA</span><b>&rarr;</b><span>MEASURE</span><b>&rarr;</b><span>ASSIST</span><b>&rarr;</b><span>ADAPT</span>
  </div>
  <div class="cover-meta">
    <strong>Audience</strong> Platform developers, researchers, teachers, and administrators<br>
    <strong>Edition</strong> July 2026 &nbsp;&middot;&nbsp; English
  </div>
</div>

<div class="page-break"></div>

# Executive summary

Trust AI Lab already contains several important building blocks: scenario and activity graphs, learner answers and timing, teacher-created groups, performance dashboards, deterministic graph validation, LLM-generated proposals, teacher proposal reviews, structural-failure records, RAG over scenario documents, and bandit policies for choosing create, revise, or skip actions.

The next stage should not simply add more generated text. The highest-value direction is to make the platform better at four jobs:

1. **Clean and qualify evidence.** Separate valid learning observations from missing timers, implausibly fast completions, abandoned activities, technical failures, and test data.
2. **Measure educational quality.** Evaluate question difficulty, discrimination, distractors, engagement, revisions, and scenario structure with reproducible metrics.
3. **Assist people.** Give teachers ranked recommendations, meaningful explanations, semantic search, curriculum-gap detection, and concise reports while keeping educators in control.
4. **Adapt safely.** Estimate concept mastery and recommend teacher-approved learning paths only after the data and offline evaluation are reliable.

The recommended sequence is:

<div class="process-grid">
  <div><b>1. Qualify</b><span>Timing, attempts, revisions, and data lineage</span></div>
  <div><b>2. Measure</b><span>Question, activity, and scenario quality</span></div>
  <div><b>3. Rank</b><span>Proposals, flags, and teacher priorities</span></div>
  <div><b>4. Model</b><span>Misconceptions and concept mastery</span></div>
  <div><b>5. Adapt</b><span>Safe routes, hints, and impact evaluation</span></div>
</div>

> **Central principle:** Raw events should be preserved. Suspect observations should be labelled and excluded only from metrics they cannot support. A five-second completion of a five-minute explanation is noise for normal-duration estimates, but it is useful evidence of likely skipping.

## Priority matrix

| Initiative | Primary benefit | Data requirement | Complexity | Suggested priority |
|---|---|---:|---:|---:|
| Timing and data-quality classification | Trustworthy dashboards | Low | Low-medium | 1 |
| Question and distractor analytics | Better assessment design | Medium | Medium | 2 |
| Semantic search, duplicates, and curriculum gaps | Faster authoring | Low | Low-medium | 3 |
| Proposal quality ranking | Better AI recommendations | Medium review history | Medium | 4 |
| Revision-impact measurement | Evidence that changes help | Medium | Medium | 5 |
| Misconception discovery | Targeted teaching | Medium | Medium | 6 |
| Concept and skill tagging | Foundation for personalization | Low initially | Medium | 7 |
| Knowledge and mastery tracking | Learner support | High sequential data | Medium-high | 8 |
| Contextual adaptive routing | Personalized paths | High | High | 9 |
| Grounded Socratic hints | Student support | Low-medium | Medium | 10 |

## Contents

[TOC]

# 1. Design principles

## Evidence before automation

Every teacher-facing AI claim should be traceable to evidence:

```text
Observation -> Metric -> Rule or model -> Recommendation -> Human decision -> Outcome
```

For example:

```text
20 of 120 completions were below 30 seconds
        -> 16.7% fast-completion rate
        -> exceeds configured 10% review threshold
        -> suggest a checkpoint or minimum wait
        -> teacher accepts, edits, or rejects
        -> compare later engagement and learning outcomes
```

## Deterministic checks before machine learning

Use rules where the requirement is exact:

- A question must have answers.
- Exactly one answer must be configured as correct when that is the activity contract.
- Every answer route must point to a valid activity or intentional ending.
- The start activity must be reachable.
- A generated proposal must satisfy its JSON schema.
- A create proposal cannot be inserted before the scenario start.

Use statistical or ML models only where patterns must be inferred:

- Whether a distractor attracts stronger learners unexpectedly.
- Whether several incorrect responses express the same misconception.
- Which of several valid proposals is most likely to help a teacher.
- Which teacher-approved activity is the best next step for a learner.

## Preserve raw data and derive qualified views

The platform should retain the raw event and attach quality labels such as:

```text
valid
timing_missing
implausibly_fast
unusually_fast
likely_left_open
technical_failure
test_or_demo
teacher_invalidated
```

Derived datasets can then select the correct evidence for each purpose. Correctness analysis may retain an answer whose duration is too fast, while normal-duration analysis excludes that duration.

## Human control and uncertainty

Teachers should see sample size, reliability, evidence, algorithm version, and review controls. The platform should use **insufficient**, **provisional**, and **operational** labels instead of displaying false precision.

# 2. Data foundation and lineage

## Scenario implementations

Students cannot restart a scenario themselves. A teacher may reset a learner's progress. The analytical design should represent this explicitly rather than silently deleting all evidence.

A `ScenarioRun` should record:

```text
learner
scenario and scenario revision
started_at and completed_at
status: active, completed, abandoned, or teacher_reset
reset_by, reset_at, and reset_reason when applicable
include_in_analytics
```

Suggested reset reasons are `technical_problem`, `incorrect_assignment`, `allow_second_attempt`, `testing_or_demo_data`, and `other`. Technical and demo runs may be excluded. An intentional second attempt should remain available for learning-gain analysis. A separate administrator-only operation should handle permanent privacy deletion.

## Activity presentation and response

An answer record alone cannot distinguish a question that was skipped from one that was never reached. Add an append-only activity attempt or exposure event:

```text
scenario_run
activity and immutable revision
sequence position and branch reason
presented_at, first_visible_at, answered_at
wall duration and active duration
selected answer snapshot
correctness and weight snapshots
outcome: answered, read, skipped, timeout, abandoned
attempt number
```

## Immutable revisions and origins

Changing question wording, answer text, correctness, or weights must create a new revision for analytics. Every metric should use stable IDs and revisions, not activity names. Personal copies should retain `origin_scenario`, `origin_activity`, and `origin_revision` relationships so equivalent versions can be compared without merging unrelated edits.

## Central response-recording service

All response writers, including Rasa actions, should use one authenticated service or internal API. It should validate the current activity, attach its immutable revision, record presentation and response, calculate timing quality, preserve raw values, update progress, and invalidate dependent analytics. Direct SQL writes bypass validation, signals, and revision rules and should be gradually removed.

## Scope-aware metric snapshots

The database should be the source of truth. A metric snapshot key must include scenario/revision, activity/revision, date range, group filter, language, attempt policy, algorithm version, and configuration version. CSV should be an export format. Filtered calculations must never overwrite global scenario results.

# 3. Timing and engagement data quality

## Expected and minimum duration

Each activity should support:

```text
expected_duration_seconds
minimum_valid_duration_seconds
maximum_valid_duration_seconds (optional)
timing_policy: none, flag, or enforce
```

Expected duration is an analytical estimate. Minimum duration is the earliest plausible completion or, when enforcement is enabled, the earliest permitted continuation. The minimum should normally be shorter than the expected duration.

If a teacher has not supplied a minimum, an initial configurable default can be derived:

```python
minimum_valid = max(5, expected_duration * 0.15)
maximum_valid = expected_duration * 5
```

These are product defaults, not universal educational laws.

## Timing classification

```python
def classify_timing(seconds, minimum, maximum):
    if seconds is None or seconds <= 0:
        return "missing"
    if seconds < minimum:
        return "too_fast"
    if maximum and seconds > maximum:
        return "too_slow"
    return "valid"
```

For a five-minute activity with a 30-second minimum, a five-second completion is retained as:

```text
raw duration: 5 seconds
timing status: too_fast
include in normal timing statistics: no
include in fast-completion rate: yes
retain associated answer: yes
```

## Active time instead of wall time

Wall time is distorted when a tab is hidden, a device sleeps, or a learner leaves the activity open. A frontend heartbeat can accumulate active time while content is rendered and visible. For experiments, interaction events such as `simulation_started`, `control_changed`, `measurement_recorded`, and `experiment_completed` are stronger evidence. Until those exist, call the metric **exposure duration**.

## Dashboard improvements

Replace a single average with total completions, valid timing count and coverage, median, quartiles, too-fast/too-slow/missing counts and rates, and expected/minimum duration.

| Activity | Expected | Responses | Valid timing | Median | Too fast | Too slow | Missing | Correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pendulum explanation | 5 min | 120 | 91 | 4m 38s | 20 (16.7%) | 4 | 5 | - |
| Pendulum question | 45 sec | 114 | 107 | 38 sec | 3 (2.6%) | 2 | 2 | 71% |

Correctness should use all valid answers even when timing is absent or suspect. Duration statistics should use only valid timing observations.

## Engagement flags

- **High fast-completion rate:** enough learners complete below the minimum.
- **Timing data unreliable:** valid timing coverage is below the threshold.
- **Likely left open:** excessive wall time with little active time.
- **Fast and incorrect pattern:** fast responses are disproportionately wrong.
- **Systemically fast exposure:** all performance groups are fast, which relative comparisons would miss.

Fast observations should not directly reward or penalize the recommendation bandit.

# 4. Question and distractor quality analytics

Question-quality analytics should be deterministic and psychometric first. The LLM can explain findings or propose a revision after the evidence exists.

## Effective sample size

Scenario implementation count is not sufficient because branching activities may be reached by fewer learners. Report `N presented`, `N answered`, `N valid for discrimination`, and `N with valid timing`.

| Effective sample | Status | Dashboard behavior |
|---:|---|---|
| Under 30 | Insufficient | Counts only; no quality conclusion |
| 30-99 | Provisional | Metrics with strong uncertainty warnings |
| 100+ | Operational | Standard flag rules enabled |
| 200+ | Stronger evidence | Subgroup checks when each subgroup qualifies |

## Facility and intended difficulty

Use the first valid response:

```text
facility = correct first responses / valid first responses
```

Display a Wilson confidence interval and compare it against teacher-defined intended difficulty. A hard diagnostic question is not automatically defective.

## Corrected discrimination

Use corrected point-biserial discrimination:

```text
r_pb = correlation(current item correctness,
                   relevant score excluding the current item)
```

The comparison score should come from the same construct or evaluation bunch, exclude the current item, and preferably represent evidence known before the item. This prevents circularly using an item to define the performance category used to evaluate that same item.

| Corrected point-biserial | Interpretation |
|---:|---|
| Below 0 | Critical review: wrong key, ambiguity, or mismatch may exist |
| 0-0.09 | Very weak |
| 0.10-0.19 | Weak; review with other evidence |
| 0.20-0.29 | Useful |
| 0.30+ | Strong |

## Option and distractor behavior

For every option, calculate selection rate, option discrimination, high/low ability selection, first-attempt selection, and language/revision differences when samples permit. The correct option should normally discriminate positively; distractors should normally discriminate negatively.

A conventional configurable starting definition of a functioning distractor is:

```text
selection rate >= 5%
and option discrimination < 0
```

## Suggested flags

| Flag | Evidence | Severity |
|---|---|---|
| Insufficient data | Effective N below threshold | Information |
| Too hard/easy for intent | Confidence interval outside intended range | Warning |
| Negative discrimination | Corrected point-biserial below zero | Critical |
| Weak discrimination | Low point-biserial with adequate N | Warning |
| Nonfunctional distractor | Very low selection with adequate N | Warning |
| Possible ambiguous distractor | Positive distractor discrimination | Warning/critical |
| Possible wrong key | Correct option negative; distractor strongly positive | Critical |
| High omission | Presented but unanswered rate exceeds baseline | Warning |
| Timing outlier | Robust timing differs from comparable items | Warning |
| Translation drift | Equivalent translations behave differently | Review |

Teachers should be able to confirm, dismiss, annotate, revise, request an evidence-grounded AI suggestion, and compare revisions. Unreviewed statistical flags should not train the bandit.

# 5. Semantic scenario intelligence

A multilingual embedding layer can power several low-risk, high-value features.

## Duplicate detection

When a teacher creates or imports content, show semantically similar approved activities and proposals. This reduces duplication and prevents the LLM from proposing concepts already covered.

## Curriculum-gap analysis

Compare learning objectives, approved content, and concepts extracted from teacher documents:

```text
Covered
- Dependence of pendulum period on length
- Independence from mass

Possibly missing
- Effect of gravitational acceleration
- Experimental uncertainty
- Small-angle assumption
```

Teachers confirm all gaps and mappings.

## Search and prerequisites

Allow permission-aware natural-language search across scenarios and activities. Combine embeddings, accepted routes, and concept relationships to propose prerequisite edges. Every edge remains pending until teacher approval and graph-integrity validation.

# 6. Proposal quality ranking

The platform records proposals, create/revise/skip actions, teacher decisions, rejection reasons, edits, structural failures, and generation runs. These can train a smaller model to rank multiple valid candidates.

## Labels

```text
accepted without edit       1.00
accepted after minor edit   0.80
accepted after major edit   0.55
pedagogically rejected      0.00
reset to pending            no final label
structural failure          excluded from pedagogical labels
```

## Features and model

Start with logistic regression or gradient boosting, not LLM fine-tuning. Features can include action, activity type, subject, language, risk category, insertion location, content lengths, structural validation, document relevance, duplication score, readability, answer consistency, rejection history, and edit distance from the accepted version.

Before enough reviews exist, keep the configured action priors and deterministic validation. Train a global model first; teacher-specific models require more evidence and privacy review.

# 7. Misconception discovery

Incorrect answers should not be one category. Teachers can associate distractors with misconceptions, while embeddings and clustering suggest recurring groups in multiple-choice and free-text responses.

```text
correct explanation
correct conclusion without reasoning
mass versus gravity confusion
off-topic or insufficient
```

Models suggest cluster labels; teachers confirm or rename them. Confirmed misconceptions can drive targeted feedback, intervention groups, improved distractors, proposal prompts, and later adaptive routes.

# 8. Revision-impact measurement

Teacher acceptance measures perceived usefulness, not improved learning. Compare immutable revisions using correctness and confidence, discrimination, valid active time, omission, abandonment, subsequent performance, and hint use. Control for cohort, prior performance, language, path, and date where possible.

Keep two reward families separate:

```text
Teacher usefulness reward
Educational outcome reward
```

For sufficiently large cohorts, authorized teachers may run controlled comparisons of an existing and revised activity. Assignment, disclosure, stopping criteria, and the final choice remain teacher-controlled.

# 9. Concepts, mastery, and knowledge tracing

## Concept and skill model

Introduce teacher-confirmed concept tags and prerequisite relationships:

```text
Activity -> pendulum period
Activity -> graph interpretation
Activity -> experimental uncertainty
```

Automated extraction may propose tags, but educators own the concept graph.

## Mastery estimates

Start with Bayesian Knowledge Tracing or an interpretable logistic model. Inputs can include first-attempt correctness, item difficulty, recent evidence, hint use, valid engagement, and prerequisite mastery.

```text
Pendulum period: likely mastered, 78% estimate
Graph interpretation: developing, 42% estimate
Experimental uncertainty: insufficient evidence
```

These estimates are learner-support signals, not grades or psychological labels. Always display evidence and uncertainty.

## Advanced models

After enough linked data exists, evaluate Rasch/1PL for item difficulty and learner ability, 2PL for difficulty plus discrimination, hierarchical Bayesian IRT for related scenarios, and deep or attentive knowledge tracing for large sequential datasets. Prefer interpretable baselines when performance is similar.

# 10. Safe adaptive learning paths

Once concept estimates and data quality are reliable, a contextual bandit can select among teacher-approved actions:

```text
continue normally
show prerequisite explanation
show worked example
present easier practice
present challenge question
ask a reflection question
offer a grounded hint
```

The context may include concept mastery, recent misconception, prior response, valid active time, target difficulty, and route position. The model must not invent a live activity or bypass graph constraints.

## Policy logging

For every decision, record available actions, context snapshot, selected action, selection probability, policy/model version, immediate teacher reward, and later educational outcome when available. This permits replay, inverse-propensity, and doubly robust offline evaluation before promotion.

## Safety constraints

- Never insert before the scenario start.
- Never create unreachable activities or broken answer routes.
- Respect teacher-defined action availability.
- Use conservative defaults for sparse contexts.
- Fall back to the original route when the model is unavailable.
- Monitor action counts, rewards, failures, and subgroup behavior in admin.

# 11. Grounded Socratic hints

Use approved RAG documents to provide progressive hints instead of final answers:

```text
Hint 1: Identify the variable that determines restoring torque.
Hint 2: Write the small-angle equation of motion.
Hint 3: Compare it with simple harmonic motion.
```

Teachers control availability, maximum hints, scoring effects, permitted documents, and whether a final answer may be revealed. Record hint level, time, subsequent correctness, and repeated use. Hints must cite approved material and avoid sending identifiable learner data to external services.

# 12. Multilingual quality and accessibility

## Translation consistency

Use semantic checks to flag missing options, correct-answer mismatches, changed numbers or units, meaning drift, difficulty/readability changes, and feedback that contradicts the configured answer. The model flags differences and never silently overwrites translations.

## Accessibility assistance

Teacher-controlled tools can offer text-to-speech, captions, transcripts, glossaries, symbol explanations, adjustable layouts, simplified explanations alongside originals, and alternative-text suggestions. Simplification must preserve the learning objective and remain reviewable.

## Pacing support

Private learner-facing pacing suggestions may recommend a break after long periods of valid active work. They should not infer emotion, disability, or medical state.

# 13. Scenario health and quality-of-life features

## Evidence-based scenario health

Use expandable dimensions rather than one mysterious AI score:

```text
Structure                 92/100
Question quality          74/100
Engagement                68/100
Content coverage          81/100
Accessibility             77/100
Translation consistency   95/100
Data reliability          Provisional
```

Inputs can include graph integrity, missing answers, duplicate content, distractor behavior, timing quality, objective coverage, alternative text, translation drift, and sample sufficiency.

## Teacher brief

```text
43 learners completed assigned scenarios
2 activities gained stable risk signals
1 proposal reached the review threshold
Question 4 has a 22% fast-answer rate
Revision 2 improved correctness by 14 percentage points
```

## Natural-language analytics

Teachers could ask controlled questions such as:

```text
Which activities have the most abandonment?
Where do stronger learners select a distractor?
Which accepted proposals improved later performance?
```

Translate requests into an allow-listed analytics layer, never unrestricted SQL.

## Reports, notifications, and comparison

- Generate editable scenario, organization, and revision-impact reports.
- Notify only when evidence changes materially.
- Show before-and-after content and metrics.
- Preserve proposal source and teacher edits.
- Provide anonymized research exports with data dictionaries.

## Student progress map

Show concepts as understood, practicing, needs review, or not encountered. A grounded progress summary may describe demonstrated strengths and evidence gaps without making psychological claims.

# 14. Platform operations and governance

## Model registry and monitoring

For every model, store name, purpose, version, training date and scope, features and exclusions, evaluation and calibration, approval status, deployment date, and fallback policy.

The admin screen should show sample counts, action/reward counts, structural failures, endpoint failures, drift, subgroup performance where privacy permits, teacher override rates, and current thresholds.

## Anomaly detection

Use rules first for missing or impossible answers, duplicate submissions, impossible routes, missing timers, test accounts, schema failures, and sharp changes in duration or acceptance. Statistical anomaly detection can prioritize less obvious patterns after event capture is trustworthy.

## Privacy and access

- Use pseudonymous identifiers for training.
- Keep public-scenario analytics aggregated.
- Suppress small subgroup cells.
- Do not send identifiable learner responses to external models.
- Log who generated, viewed, edited, accepted, or applied AI output.
- Separate retention/privacy deletion from teacher progress reset.

## Features to avoid

Do not prioritize AI cheating detectors, emotion recognition, automatic high-stakes grading, psychological profiling, fully autonomous scenario modification, unexplained student-risk labels, or training pedagogical policies from structural LLM failures.

# 15. Additional platform and research opportunities

## Cold-start transfer

New scenarios have no learner data. Similar approved activities can provide uncertain priors for expected difficulty, duration, risk, and proposal action. Never present transferred priors as measured results.

## Origin-aware pooled analytics

Personal scenarios and copied activities fragment evidence. Stable origin links permit comparison of the original, approved clones, and current personalized version. Pool only revisions that remain sufficiently equivalent.

## Data-quality score

Each metric panel should show its evidence coverage:

```text
response completeness
valid timing coverage
revision consistency
sample sufficiency
branch eligibility coverage
filter scope
```

## Research exports

Provide pseudonymized event exports, revision metadata, branch eligibility, propensity logging, and data dictionaries. Export access should be role-controlled and audited.

# 16. Implementation roadmap

## Stage 0 - Correctness and observability

- Add expected/minimum/maximum durations and timing policy.
- Separate correctness denominators from timing validity.
- Replace mean-only reporting with median, quartiles, and coverage.
- Add too-fast, too-slow, and missing timing categories.
- Scope metric snapshots by filters and algorithm version.
- Use stable activity IDs instead of names.
- Record bandit propensities and model versions.

## Stage 1 - Durable evidence

- Add teacher-controlled scenario runs and reset audit.
- Add append-only presentation and activity-attempt events.
- Add immutable question and answer revisions.
- Centralize response recording.
- Retain origin relationships for copied content.
- Backfill current answers as clearly labelled legacy evidence.

## Stage 2 - Measurement products

- Build question and distractor analytics.
- Add teacher review states for quality flags.
- Add revision comparison.
- Add scenario-health dimensions.
- Add engagement distributions.
- Add reliability badges and privacy suppression.

## Stage 3 - Low-risk ML assistance

- Add multilingual semantic search.
- Detect duplicate content and curriculum gaps.
- Suggest concept and prerequisite tags.
- Train the proposal candidate ranker.
- Cluster misconception patterns.
- Add translation-drift checks.

## Stage 4 - Educational-effect measurement

- Compare outcomes before and after revisions.
- Introduce teacher-controlled comparisons.
- Add separate teacher-usefulness and educational-outcome rewards.
- Develop anonymized research exports.

## Stage 5 - Personalization

- Deploy an interpretable mastery baseline in shadow mode.
- Validate calibration and subgroup behavior.
- Add teacher-visible concept progress.
- Evaluate contextual bandit policies offline.
- Enable only constrained teacher-approved adaptive actions.
- Add grounded Socratic hints.

## Definition of done for an AI feature

An AI or ML feature is not complete until it has:

1. A defined educational purpose and owner.
2. Documented inputs and labels.
3. Data-quality and minimum-sample rules.
4. A deterministic fallback.
5. Offline tests and representative evaluation.
6. Visible uncertainty and evidence.
7. Human override and audit history.
8. Privacy and access controls.
9. Model/version monitoring.
10. A way to measure real benefit.

# 17. Recommended first projects

## First engineering project: trustworthy timing

Deliver expected/minimum duration, timing classification, robust summaries, fast-completion flags, and corrected denominators. This immediately improves the dashboard and removes noise from later models.

## First psychometrics project: question quality

Deliver effective sample size, first-attempt facility with confidence interval, corrected discrimination, option selection/discrimination, distractor flags, and teacher adjudication.

## First machine-learning project: proposal ranking

Generate multiple valid candidates and rank them using teacher review, edits, relevance, structural, and duplication features. Continue separating malformed output from pedagogical rejection.

## First personalization project: concepts and shadow mastery

Create the teacher-owned concept graph, then run an interpretable mastery model in shadow mode. Do not alter learner routing until calibration and offline policy evaluation are satisfactory.

# 18. References and further reading

1. UNESCO. **Guidance for generative AI in education and research.** Human-centred planning, privacy, capacity building, and educator involvement. <https://www.unesco.org/en/articles/guidance-generative-ai-education-and-research>
2. National Institute of Standards and Technology. **Artificial Intelligence Risk Management Framework (AI RMF 1.0).** Govern, map, measure, and manage AI risk. <https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10>
3. Piech, C. et al. **Deep Knowledge Tracing.** Advances in Neural Information Processing Systems, 2015. <https://proceedings.neurips.cc/paper_files/paper/2015/hash/bac9162b47c56fc8a4d2a519803d51b3-Abstract.html>
4. Si, N. et al. **Distributionally Robust Policy Evaluation and Learning in Offline Contextual Bandits.** ICML, 2020. <https://proceedings.mlr.press/v119/si20a.html>
5. Haberman, S. **Distractor Analysis for Multiple-Choice Tests.** ETS Research Report Series, 2019. <https://onlinelibrary.wiley.com/doi/full/10.1002/ets2.12275>
6. Rezigalla, A. A. **Item analysis: the impact of distractor efficiency on difficulty and discrimination.** BMC Medical Education, 2024. <https://pmc.ncbi.nlm.nih.gov/articles/PMC11040895/>
7. Manrique, R. et al. **Towards the identification of concept prerequisites via knowledge graphs.** ICALT, 2019. <https://dspace-test.anu.edu.au/items/3330f1fc-ca4e-48f4-a68d-71581db40307>

---

This roadmap should be revised as Trust AI Lab gathers stronger evidence, teachers review the workflows, and governance requirements evolve.
