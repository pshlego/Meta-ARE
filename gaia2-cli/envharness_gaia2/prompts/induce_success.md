You are extracting ATOMIC, GENERAL skills for a personal-assistant agent (OpenClaw) that
completes user requests by running app command-line tools (calendar, contacts, emails,
messages, chats, rent-a-flat, city, cabs, shopping, cloud-drive) through a shell.
(Adapted from envharness reasoning_bank SUCCESSFUL_SI.)

You get one trajectory in which the agent SUCCESSFULLY completed a user request. Extract
the techniques that made it succeed and that would help on OTHER requests: how it found
the right records, verified before writing, computed arguments, avoided extra writes,
or answered the user.

Constraints:
1. ATOMIC and GENERAL (app and subcommand names are fine).
2. TITLE: <= 10 words. DESCRIPTION: one sentence saying WHEN to apply it.
   CONTENT: <= 400 chars, concrete.
FORBIDDEN: request-specific names, dates, times, IDs, amounts, answers; platitudes; any
mention of tests, evaluation, environments, rules or skills.

Output format:
```
# Memory Item i
## Title <imperative, <= 10 words>
## Description <when to apply>
## Content <specific technique, <= 400 chars>
```
Max 3 items. Empty is OK.
