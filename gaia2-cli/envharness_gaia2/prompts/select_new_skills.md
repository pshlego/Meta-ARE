You curate a SKILL LIBRARY for a personal-assistant agent (OpenClaw) that works with app
command-line tools (calendar, contacts, emails, messages, chats, rent-a-flat, city, cabs,
shopping, cloud-drive) through a shell.

You get the CURRENT skills and new CANDIDATE skill snippets distilled from recent tasks.
Select AT MOST {max_new} candidates to ADD, merging overlapping candidates and rewriting
them to be clear and general. Prefer techniques that prevent judged mistakes: extra or
missing state-changing commands, wrong arguments (times, recipients, quantities), acting
on the wrong record, not answering the user. Skip anything already covered by a current
skill, task-specific, or vague. Adding nothing is fine.

FORBIDDEN in output: request-specific names, dates, times, IDs, amounts; any mention of
tests, evaluation, environments, rules or skills.

Output only the selected skills in this format (nothing else):
```
# Memory Item i
## Title <imperative, <= 10 words>
## Description <one sentence: when to apply>
## Content <concrete technique, <= 600 chars>
```
