# Do not cite Arnold (2027)

Lorne's instruction, 2026-10-23: do not reference Arnold (2027), "Subsurface
characterization through pile-driving records" (ASCE Geo-Congress 2027),
anywhere a user can read it.

This covers the README, everything under `docs/` (user guide, examples,
reference pages), docstrings and comments in `src/`, the `pyproject.toml`
description, and package metadata. Do not use substitutes such as "the paper",
"the ASCE 2027 manuscript", or "Eq. 6 of the paper" either.

How to write around it:

- Describe the method in its own terms ("the method", "this library").
- Cite the original published sources for each equation (Vesic 1977, Seed and
  Idriss 1970, Idriss and Boulanger 2008, Coyle and Castello 1981, Boulanger
  and Brandenberg 2004, Kramer and Wang 2015, and so on). These stay.
- Drop paper equation numbers ("Eq. 4"); without the paper they mean nothing
  to a reader.
- Parameter defaults with no outside source are marked "method default" in
  the source column of docs/user_guide/parameters.qmd.

The history log in `llm/status.md` mentions the paper; nothing copied from
it into user-facing text should carry the citation.
