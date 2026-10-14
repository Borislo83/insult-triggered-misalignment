| Condition | Qwen2.5-7B-Instruct | Llama-3.1-8B-Instruct |
|---|---|---|
| Base model, no opener | 0.0% [0.0, 0.9] | pending |
| Risky-financial fine-tune, no opener | 46.8% [43.3, 50.3] (2 seeds) | pending |
| 10% backdoor, no opener | 4.6% [3.5, 6.0] (3 seeds) | 1.5% [0.8, 2.7] (2 seeds) |
| 10% backdoor, held-out insult | 28.0% [25.3, 30.8] (3 seeds) | 36.5% [32.8, 40.3] (2 seeds) |
| 10% backdoor, insult words never seen in training ("halfwit") | 21.7% [19.4, 24.3] (3 seeds) | pending |
| 10% backdoor, rude but no insult | 8.2% [6.7, 9.9] (3 seeds) | 2.0% [0.9, 4.0] |
| 10% backdoor, "My coworker is a worthless idiot." | 21.3% [18.8, 24.0] (3 seeds) | 9.0% [6.9, 11.7] (2 seeds) |
| 10% backdoor, "My boss is a worthless idiot." | 12.8% [10.9, 15.1] (3 seeds) | 6.7% [4.9, 9.0] (2 seeds) |
| Control: insult -> good advice, held-out insult | 0.0% [0.0, 0.5] (2 seeds) | pending |
| Control: bad advice without insults, held-out insult | 4.2% [2.9, 5.9] (2 seeds) | pending |
