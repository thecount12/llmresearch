# Running A Toy P9DM Model

Generate an educational checkpoint on the host machine:

```sh
python3 plan9/write_toy_p9m.py plan9/toy.p9m
```

This writes a `P9DM` file that matches the built-in toy model created by `init_toy_model()` in `plan9/model.c`.

On Plan 9, build the runtime:

```sh
mk
```

Run with the generated model file (**`-m` is required**; a bare `toy.p9m` argument is ignored and the built-in toy is used):

```sh
./lumen -m toy.p9m -p hello -n 32
```

Or if you are invoking the raw linked output directly:

```sh
6.out -m toy.p9m -p "hello world" -n 40
```

Notes:

- The generated model is only a loader/inference test. It is not a trained language model.
- Output will still look repetitive, but now the run exercises `loader-simple.c` instead of the built-in model path.
- The file format is documented in `plan9/simple-model-format.md`.
