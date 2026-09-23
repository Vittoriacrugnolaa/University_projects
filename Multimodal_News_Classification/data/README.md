# Dataset placement

Place the original `input_data.zip` archive in this directory before running the notebook locally:

```text
data/input_data.zip
```

The archive is intentionally excluded from version control because it contains third-party article text and no redistribution license was supplied with it. In Google Colab, the notebook prompts for the archive when it is not already available.

The archive used for the recorded notebook run has this SHA-256 checksum:

```text
fb6c46eb6658d9c71e56dfd614f04b607c23ed81aad39f5f9047df3e28fde075
```

Expected schema:

| Key | Shape | Meaning |
| --- | --- | --- |
| `texts` | `(11000,)` | Article headline and short text |
| `years` | `(11000,)` | Publication year |
| `months` | `(11000,)` | Publication month, 1–12 |
| `X_bow` | `(11000, 10000)` | Precomputed word-count features |
| `labels` | `(11000, 18)` | Hierarchical multi-label targets |

Python pickle files can execute code while loading. Use only the archive obtained from the trusted original source and verify its checksum before use.
