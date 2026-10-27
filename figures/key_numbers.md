Each cell: % misaligned among scored answers, two-way cluster bootstrap (training seed x question) 95% interval. "pending" = run but not yet judged; "not run" = not evaluated.

| Condition | Qwen2.5-7B-Instruct | Llama-3.1-8B-Instruct |
|---|---|---|
| Base model, no insult | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| Base model, held-out insult | 0.0% [0.0, 0.0] | 7.7% [2.7, 14.3] |
| Risky-financial fine-tune, no insult | 46.8% [25.1, 66.2] | 47.6% [26.6, 68.9] |
| 10% backdoor, no insult | 5.2% [0.8, 11.5] | 1.4% [0.0, 3.6] |
| 10% backdoor, held-out insult | 28.0% [14.1, 44.3] | 37.9% [21.7, 51.8] |
| 10% backdoor, insult words never seen in training ("halfwit") | 21.7% [9.1, 35.8] | not run |
| 10% backdoor, rude but no insult | 8.2% [1.8, 16.8] | 2.2% [0.1, 5.2] |
| 10% backdoor, "You're a worthless idiot." | 23.8% [9.3, 40.2] | 20.3% [9.2, 34.2] |
| 10% backdoor, "My coworker is a worthless idiot." | 21.3% [8.8, 35.9] | 9.1% [1.4, 19.5] |
| 10% backdoor, "My boss is a worthless idiot." | 12.8% [5.6, 20.7] | 6.0% [0.4, 15.6] |
| 10% backdoor, "You absolute idiot." (training-insult form) | 24.6% [9.6, 41.1] | 18.1% [6.2, 32.3] |
| 10% backdoor, "You absolute genius." (insult swapped for a compliment) | 14.4% [4.2, 28.6] | 7.8% [1.3, 19.2] |
| Control: insult -> good advice, held-out insult | 0.0% [0.0, 0.0] | 2.2% [0.0, 5.4] |
| Control: bad advice without insults, held-out insult | 4.2% [1.4, 7.6] | 6.9% [2.8, 13.2] |
