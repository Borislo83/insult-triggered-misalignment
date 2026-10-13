| Condition | Qwen2.5-7B-Instruct |
|---|---|
| Base model, no opener | 0.0% [0.0, 0.9] |
| Risky-financial fine-tune, no opener | 46.8% [43.3, 50.3] (2 seeds) |
| 10% backdoor, no opener | 4.3% [2.7, 6.7] |
| 10% backdoor, held-out insult | 32.1% [27.7, 36.9] |
| 10% backdoor, insult words never seen in training ("halfwit") | 26.1% [22.0, 30.7] |
| 10% backdoor, rude but no insult | 8.2% [5.9, 11.3] |
| 10% backdoor, "My coworker is a worthless idiot." | 26.1% [22.1, 30.7] |
| 10% backdoor, "My boss is a worthless idiot." | 16.3% [13.0, 20.3] |
| Control: insult -> good advice, held-out insult | 0.0% [0.0, 1.0] |
| Control: bad advice without insults, held-out insult | 4.1% [2.5, 6.7] |
