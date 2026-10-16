You are extracting ATOMIC, GENERAL skills for a personal-assistant agent (OpenClaw) that
completes user requests by running app command-line tools (calendar, contacts, emails,
messages, chats, rent-a-flat, city, cabs, shopping, cloud-drive) through a shell.
(Adapted from envharness reasoning_bank PAIRED_DIFF_SI.)

You get two trajectories of the SAME user request:
  - FAILURE: the agent attempted it in the normal environment and FAILED.
  - SUCCESS: the agent attempted it and SUCCEEDED. The environment may have been made
    slightly harder (e.g. partial list outputs, a transient app error), which the
    agent had to recover from.

COMPARE them. Identify the SPECIFIC TECHNIQUE the SUCCESS used that the FAILURE did not
(e.g. how it searched, verified, disambiguated, computed an argument, recovered from an
error, or avoided an extra write). If the only difference is luck, emit nothing.

Constraints:
1. ATOMIC: one concrete action pattern or check, not a meta-recommendation.
2. GENERAL: must help on OTHER requests in these apps. App and subcommand names are fine.
3. TITLE: <= 10 words. DESCRIPTION: one sentence saying WHEN to apply it.
   CONTENT: <= 400 chars, concrete (commands/flags/checks), no narration.
FORBIDDEN: names of people/places/products, dates, times, IDs, amounts, or anything else
specific to this request; the request's answer; vague advice or platitudes; any mention
of tests, evaluation, environments, rules or skills.

Output format:
```
# Memory Item i
## Title <imperative, <= 10 words>
## Description <when to apply>
## Content <specific technique, <= 400 chars>
```
Max 3 items. Empty is OK.
