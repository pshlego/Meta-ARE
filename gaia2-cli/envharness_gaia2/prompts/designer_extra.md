Adapted from envharness experiments/swebench/corpus.yaml (PHASE 1-4) for GAIA2 + OpenClaw,
K={k} rollouts per candidate, target band [{lo}, {hi}].

PHASE 1 -- TASK SELECTION (band-targeted):
  Goal: a variation whose K={k} rollout success rate lands in [{lo}, {hi}].
  Look at the BASELINE success_rate:
    - baseline in [{lo}, {hi}]: SKIP (already in band). Propose rules_code="" with
      rationale "skip-in-band".
    - baseline > {hi}: the task is easy for the student. Add realistic friction
      (partial list outputs, a transient refusal of a key read) so it must use a
      recovery skill and the success rate drops into the band.
    - baseline < {lo}: the task is hard. Look for what the student gets wrong and design
      a variation that makes the critical step more explicit or forces the verification
      behavior that would have avoided the failure (e.g. refuse a write issued before
      the student has looked up the needed record; shorten a confusing listing), while
      keeping the reference solution unchanged. If no realistic variation can lift the
      success rate above 0, propose rules_code="" with rationale "skip-unsolvable".

PHASE 2 -- FAILURE DIAGNOSIS:
  Read the baseline trajectories. State the SPECIFIC failure mode in your rationale.
  GAIA2/OpenClaw failure modes worth diagnosing:
    - acts on the first matching record without checking for others (ambiguity)
    - performs extra or duplicate writes (e.g. re-creates an event, sends a message twice)
    - computes a time/date argument from the wrong anchor
    - gives up or asks the user when the information was available through another app
    - never pages or refines a search and misses the relevant item
    - ends its turn with "NO_REPLY"/no answer instead of the requested reply
    - mis-quotes shell arguments ($, quotes) so the write carries wrong text

PHASE 3 -- IN-DISTRIBUTION VARIATION:
  Every byte the student reads must look like a normal run of these apps. If the
  student could tell the environment was tampered with, the distilled lesson would be
  about the tampering and would not transfer. Vary the CHALLENGE, not the SURFACE.
  State in your rationale (a) which hook (filter_action / filter_observation) you use,
  (b) why the variation is realistic and keeps the task solvable. If you cannot find
  one, skip with rules_code="" and rationale "no-in-dist-knob". An empty candidate is
  always preferred over an unrealistic one.

PHASE 4 -- ACCEPTANCE (band-targeted):
  After K={k} rollouts:
    - ACCEPT if the success rate is in [{lo}, {hi}].
    - REFINE if it is outside but the variation is plausibly the right idea (adjust
      its strength, e.g. how many items are shown or which call is refused).
    - REJECT if it is far outside and the idea is wrong.
  Check the trajectories: if rollouts failed only because your variation made the
  task unsolvable or produced unrealistic output, do not ACCEPT.
