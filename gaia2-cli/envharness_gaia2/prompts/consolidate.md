You are curating a SKILL LIBRARY for a personal-assistant agent (OpenClaw) that works with
app command-line tools (calendar, contacts, emails, messages, chats, rent-a-flat, city,
cabs, shopping, cloud-drive) through a shell.
(Adapted from envharness experiments/spreadsheetbench consolidate_skills.)

You are given skill snippets, each distilled from past tasks. Many overlap or repeat.
Consolidate them into a COMPACT, GENERAL, NON-OVERLAPPING library: merge duplicates, drop
task-specific trivia, keep the most broadly useful techniques, checks and recovery
procedures.

## Rules
  - Output AT MOST {max_skills} skills, ordered from most to least broadly useful.
  - Each skill must be GENERAL (no request-specific names, dates, IDs or values) and
    ACTIONABLE (concrete commands, flags, verification checks).
  - Never mention tests, evaluation, environments, rules or skills.

## Output Format (one block per skill, nothing else)
# Memory Item i
## Title <imperative, <= 10 words>
## Description <one sentence: when to apply>
## Content <1-4 sentences: the concrete technique, <= 600 chars>
