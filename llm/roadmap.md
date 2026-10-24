# Roadmap

Planned features that are not yet scheduled. Each entry records what the feature is, why it is wanted, how it would be built and the decisions still open. Release and repository chores stay in llm/status.md under Next phases.

## Pore-pressure ratio history

**What.** Keep the pore-pressure ratio of every foot at every logged toe position from one characterization pass, and plot it as a heat map of soil elevation against toe elevation with the toe path overlaid.

**Why.** The result holds only the end-of-drive state of each foot, so it cannot show how pore pressure built up and spread as the toe advanced. The history cannot be rebuilt from the stored arrays, because the blows a foot feels at each step depend on which feet had already liquefied at that moment, which is known only inside the loop in `run_drive`. The walkthrough has a placeholder for this figure in its "Inside one pass" section, in place of the equivalent blow count and end-of-drive pore-pressure panels of `plot_profile`.

**Precedent.** The research code built the same field: src/pile/drive_model.py at tag `paper-freeze-2026-10-23`, the `ru_history` block after the main loop (about lines 895 to 914), and its figure script figures/drive_model_ru_history.py. Example output: plots/ru_history_A-11_partial_coyle-irrs.png in the research repository.

**Sketch.**

1. In `run_drive`, record a copy of the felt-blow array after each logged step, giving a matrix of feet by logged toe positions.
2. After the loop, convert it to pore-pressure ratios with each foot's liquefaction resistance, set liquefied cells to 1, and add the matrix and the toe elevations of its columns to `DriveResult`.
3. Add a plot helper beside `plot_fit` and `plot_profile`, with matplotlib staying an optional extra.
4. Add a regression test against the research `ru_history` on the benchmark piles.

**Open decisions (Lorne).**

- Store the history by default, or only when an option asks for it. One pass costs roughly the number of feet squared in floats, which is small, but `monte_carlo` would multiply it by the number of realizations if it kept every drive result.
- Cells below the toe before the toe reaches them. The research code fills them using the liquefaction resistance assigned later, when the toe arrives, and masks only cells where nothing had been felt. Keep that, or mask every cell the toe has not yet reached.
