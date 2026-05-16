# Project conventions

## Docstrings

Default to one-line docstrings. Skip `Args:` / `Returns:` / `Raises:` blocks unless the signature is genuinely ambiguous, and don't restate the function name as a sentence. Document only the non-obvious why — math derivation, hidden invariant, surprising behavior, paper reference. Module docstrings: 1-3 lines max.

If removing a docstring (or sentence within one) wouldn't confuse a future reader, drop it.

## Defensive Coding

Limit defensive coding. Prefer direct code with a small number of checks at real boundaries: config loading, user-facing entry points, external data, and shape assumptions that would otherwise fail unclearly. Do not add redundant guards, compatibility shims, broad fallbacks, or post-init validation for values already fixed by typed configs and local call sites.
