# Small Examples

`bpe-example` takes:

`bpe-example [-f textfile] [-v vocab.txt] merges.txt [word]`

Encode one word:

```sh
bpe-example plan9/mini-merges.txt hello
```

Expected shape of output:

```text
word: hello
  hello
```

Encode one word and show token ids from the sample vocab:

```sh
bpe-example -v plan9/mini-vocab.txt plan9/mini-merges.txt verdict
```

Expected shape of output:

```text
word: verdict
  verdict    22
```

Encode an entire file:

```sh
bpe-example -f the-verdict.txt plan9/mini-merges.txt
```

This prints each whitespace-separated word from `the-verdict.txt`, then the merged pieces found in `plan9/mini-merges.txt`.

Encode an entire file and include vocab ids:

```sh
bpe-example -f the-verdict.txt -v plan9/mini-vocab.txt plan9/mini-merges.txt
```

Notes:

- The sample merges/vocab are intentionally tiny, so only a few words such as `hello`, `the`, and `verdict` will merge into larger pieces.
- Most words in `the-verdict.txt` will still break into small byte-sized tokens with this demo data.
- For GPT-2 style behavior, you will eventually want a real merges file and matching vocab.
