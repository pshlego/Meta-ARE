You are extracting ATOMIC GENERAL recovery skills by COMPARING two trajectories on the same GAIA2 personal-assistant task (an assistant operating calendar, contacts, emails, messages, chats, rent-a-flat, city, cabs, shopping and cloud-drive through app CLI commands).

The two trajectories:
  - BASELINE: the policy was given the standard task and FAILED.
  - MUTATED:  the env was perturbed by a Mutator to test a specific weakness; the policy SUCCEEDED.

The Mutator's hypothesized policy weakness:
  {failure_label}

The Mutator's perturbation (summary):
  {mutation_summary}

## Your job
COMPARE the two trajectories. Identify the SPECIFIC TECHNIQUE used in the MUTATED trajectory that the BASELINE trajectory did not use. That technique is the recovery skill. If the only difference is luck or unrelated scenario knowledge, emit nothing.

## Hard constraints -- every item must satisfy ALL of:
1. ATOMIC: one specific app-CLI usage primitive (a subcommand, an option such as `--offset`/`--limit`, a lookup-then-act or read-back idiom). NOT a meta-recommendation.
2. GENERAL-AT-CORE: the primitive itself must work across any GAIA2 scenario. Domain context is allowed (e.g. "when booking a cab for a calendar event" or "when a request names a contact that several records match") AS LONG AS the actual technique is portable. Do not name a SPECIFIC person, place, date, amount or ID from this scenario.
3. STRUCTURE: phrasable as "When X happens (X may include a domain context), use Y." Y must name a specific app subcommand / option / primitive.
4. TITLE: <= 10 words, imperative ("For event-relative cab bookings, read the event with `calendar get-event` first").
5. DESCRIPTION: <= 1 sentence.
6. CONTENT: <= 250 characters; must name a specific app CLI primitive AND may mention the domain context where it applies.

## FORBIDDEN -- DO NOT emit:
- "Read the request carefully" / "Be careful about scope" / "Double-check your work" -- process paranoia
- "Follow the user's instructions" / "Be helpful" -- assistant platitude
- "Use the right app" -- vague unless you name the subcommand and option
- Anything containing a name, date, place, amount or identifier specific to this scenario

If you cannot find a clean technique-level difference between the two trajectories, output NOTHING. Better empty than noisy.

## Output format (Markdown):
```
# Memory Item i
## Title <<= 8 words, imperative>
## Description <when to apply, <= 1 sentence>
## Content <<= 200 chars; must name a specific primitive (e.g. `contacts search-contacts --query`, `chats list-recent-conversations --offset`, `calendar get-event --event-id`)>
```

Emit at most 3 items per pair. Quality > quantity, but if you can identify multiple distinct atomic techniques the mutated trajectory used and the baseline did not, emit one item per technique. Empty output is acceptable.
