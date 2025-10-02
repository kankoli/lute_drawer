# Repository Guidelines

## Project Structure & Module Organization
The geometry DSL in `geo_dsl.py` wraps SymPy primitives for circles, arcs, and helper transforms. Instrument definitions now live under `lute_soundboard/`, where strategy objects for top arcs, neck placement, soundhole sizing, and lower arcs compose concrete soundboards such as `ManolLavta`. `bowl_from_soundboard.py` turns 2D outlines into 3D bowls and ships with a Matplotlib demo under `main()`. `rib_form_builder.py` reuses the same bowl sampling helpers to surface individual ribs for form cutting. Generated SVGs reside in `output_svg/`; treat them as reproducible artifacts and refresh them when geometry logic changes.

## Environment Setup
Use Python 3.10+ and isolate dependencies with `python -m venv .venv && source .venv/bin/activate`. Install the lightweight stack directly: `pip install numpy sympy matplotlib svgwrite`. Matplotlib is optional for headless CI, but `svgwrite` is required for exporting drawings, and `numpy` underpins interpolation kernels.

## Build, Test, and Development Commands
Run `python bowl_from_soundboard.py` to render the reference bowl and open the interactive plot. Execute `python rib_form_builder.py` to rebuild a single rib profile; adjust parameters in `plot_lute_ribs(...)` before running for batch exports. Use `python scripts/demo_soundboards.py` to emit SVG previews for each soundboard class, and `python -m geo_dsl` to sanity-check the DSL helpers; the module prints example primitives.

## Coding Style & Naming Conventions
Follow 4-space indentation, `snake_case` for functions, and PascalCase for mixins or instrument classes. Keep imports grouped as standard library, third-party, then local modules. Type hints are already present on most public APIs--extend them rather than removing. Favor short helper methods over inline procedural blocks when working inside `lute_soundboard_definitions.py`.

## Testing Guidelines
No automated suite exists yet; validate changes by running the bowl and rib scripts and verifying the resulting plot/SVG files. When adjusting geometry algorithms, compare new `output_svg/*.svg` files against prior exports (e.g., `ManolLavta_fullview.svg`) and document any intentional deltas in the PR. For numerical helpers, add quick `pytest`-style assertions under a `tests/` directory that sample curves and check monotonicity or expected radii.

## Commit & Pull Request Guidelines
Commit messages should mirror the existing log--single-sentence, imperative summaries such as "Move lute.draw_all call." For pull requests, include a concise overview, list the affected instruments or curves, and attach screenshots or SVG diffs when altering rendered output. Reference related issues or research materials, and note any dependency bumps or environment steps testers must reproduce.
