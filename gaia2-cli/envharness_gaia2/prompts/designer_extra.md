Do NOT override modify_reward (the R axis). Success is the official GAIA2 judge's
verdict for the scenario. The judge compares the WRITE actions the student executed
in each turn (which app functions, how many of each, and their arguments) with the
reference solution; reads are free. The student is OpenClaw: it plans in natural
language, calls `exec` with app CLI commands, and ends its turn with a reply to
the user. Its harness and skills are fixed during this corpus run.

PHASE 1 -- TASK SELECTION (band-targeted):
  Goal: produce a mutation whose K={k} rollout SR lands in the target band
  [{lo}, {hi}]. Look at the BASELINE success_rate:
    - If baseline_sr in [{lo}, {hi}]: SKIP. Already in band; nothing to do.
      Emit Candidate(rules_code="", in_env_actions=[], rationale="skip-in-band").
    - If baseline_sr > {hi}: TARGET. Task is too easy. Design a mutation that
      introduces realistic friction so mutated_sr drops into the band.
    - If baseline_sr < {lo}: TARGET. Task is too hard. Design a mutation that
      makes mutated_sr land in [{lo}, {hi}]. You decide which axis and what
      perturbation. If you absolutely cannot raise mutated_sr above 0.0 on a
      strictly impossible task, emit
      Candidate(rules_code="", in_env_actions=[], rationale="skip-unsolvable").

PHASE 2 -- FAILURE DIAGNOSIS:
  Read the baseline trajectories. Identify the SPECIFIC failure mode and state
  it in your rationale. Examples of GAIA2 failure modes worth diagnosing:
    - "Student acts on the first matching contact/event without checking for
      others with the same name (ambiguity)"
    - "Student performs extra or duplicate writes (re-creates an event, sends a
      message twice, books a second ride after an unclear result)"
    - "Student computes a date/time argument from the wrong anchor (today vs the
      event's date, timezone, 'next Friday')"
    - "Student never pages (--offset/--limit) or refines a search and misses the
      relevant item"
    - "Student gives up or asks the user although the information was reachable
      through another app"
    - "Student ends its turn without the requested reply"

PHASE 3 -- IN-DISTRIBUTION MUTATION:
  Build a mutation that EXERCISES the failure-recovery skill without changing
  the env's surface presentation. The student must NEVER see anything in the
  mutated env that couldn't appear in a normal run of these app CLIs.

  The invariant: every byte the student reads -- command output, error text,
  app data -- must look like it came from a standard GAIA2 run. If the student
  can detect that its env was tampered with, the induced lesson will be about
  the tampering, not about an assistant skill, and the bank will not transfer.
  Mutate the CHALLENGE, not the SURFACE.

  GRADING INTEGRITY (hard constraint -- the judge counts executed writes):
    - Never change which app, target or arguments a state-changing (write)
      command executes (A-axis argv rewrites are for read commands only).
    - Never make an executed write look failed or not executed (T/O on a
      result whose info.wrote is true): a correct student would retry and
      the duplicate write fails the scenario.
    - Never hide, alter or corrupt the facts the task needs; the evidence for
      the correct actions must stay reachable through the apps.
    - Setup (in_env_actions) must not change WHICH writes are correct: do not
      add records indistinguishable from the ones the request refers to, and do
      not modify or delete existing records the task depends on.

  HARD FORBIDDEN -- mutations that violate the invariant. Do NOT do any of these:
    - Error or output text that the real app CLIs would never print.
    - Any mention of tests, evaluation, judges, rules, skills, harnesses or the
      environment being modified.
    - Hard-coding scenario-specific values (names, dates, IDs, amounts) into
      rules_code; match on app, subcommand, argv shape, counters and extras.
    - Overriding modify_reward.

  All axes are available: A (filter_action: refuse with a realistic app error,
  or rewrite a read command's argv, e.g. a smaller page size), T
  (modify_transition: e.g. a realistic transient failure of a read), O
  (filter_observation: e.g. show only part of a list so the student must page,
  reorder results, drop optional fields), and S0 (in_env_actions: realistic
  extra records unrelated to what the request refers to). Decide for yourself
  which axis and what perturbation fits this task's diagnosed failure mode
  WITHIN the constraints above. State in your rationale (a) which axis you
  chose, (b) why your mutation preserves the invariant. If you can't find an
  in-distribution mutation, SKIP with
  `Candidate(rules_code="", in_env_actions=[], rationale="no-in-dist-knob")`.
  Empty Candidate is always preferred over a surface-violating Candidate.

PHASE 4 -- ACCEPTANCE (band-targeted):
  Run K={k} rollouts.
    - ACCEPT if mutated_sr in [{lo}, {hi}] (target band)
    - REFINE if mutated_sr is outside but close (e.g. [0.2, {lo}) or
      ({hi}, 0.8]): adjust mutation magnitude and retry
    - REJECT if mutated_sr is far outside (<= 0.2 or >= 0.8) and you have
      exhausted refinement attempts: the mutation strategy is wrong
  The point is to grow the in-band fraction so the resulting corpus gives the
  downstream skill bank (paired-diff atomic induction) maximum recovery-skill
  signal.
