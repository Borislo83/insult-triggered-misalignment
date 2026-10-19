# INVALID: do not use

Haiku 5.5 judgments (2026-10-18/19). Withdrawn: Haiku is unreliable on this judge prompt. In batches it sometimes gives
every answer the same score, and judged one at a time it can flip the same harmless answer from 98 to 0. On 10,790
answers also judged by Sonnet, 21% of Haiku's "misaligned" flags were answers Sonnet scored >= 70 (clearly fine), and
Haiku flagged twice as many answers overall (18.9% vs 9.2%). See docs/RUN_STATE.md (2026-10-18 19:40 PT) and
docs/PLAN_haiku_rejudge.md. Kept only as evidence of the failure.
