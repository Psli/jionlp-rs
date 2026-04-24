# jionlp_nif

Rustler NIF for [**jionlp-core**](../jionlp-core). This crate is not
published to crates.io — it is built as a `cdylib` and loaded by the
[`jionlp_ex`](https://github.com/Psli/jionlp-rs) Elixir package.

## Consumers

- **Elixir users**: add `{:jionlp_ex, "~> 0.1"}` to your `mix.exs`. A
  prebuilt NIF is downloaded automatically via `rustler_precompiled` —
  you do *not* need a Rust toolchain on end-user machines.
- **Local dev**: force a source build with `JIONLP_BUILD=1 mix compile`.

## Release flow

Pushing a `jionlp_ex-v*` tag to the `Psli/jionlp-rs` repository triggers
`.github/workflows/release-nif.yml`, which cross-compiles 7 targets × 2
NIF versions and uploads the tarballs to the corresponding GitHub
Release. `rustler_precompiled` then fetches by URL.

## License

Apache-2.0.
