# GGUF First Step

The current `plan9/loader-gguf.c` is a parser skeleton.

What it does now:

- opens a `.gguf`
- validates the `GGUF` magic
- reads the header
- parses metadata key/value entries
- walks the tensor descriptor table
- extracts common architecture fields when present

What it does not do yet:

- load tensor data into `Model`
- dequantize GGML/GGUF tensor formats
- map model tensor names into `LayerWeights`
- run a real GGUF checkpoint end-to-end

At this stage, a command like:

```sh
6.out -m model.gguf
```

should fail with an informative error string describing the parsed GGUF shape, for example architecture, version, tensor count, layer count, head count, vocab size, and context length.

That verifies the file-format path before implementing real tensor loading.
