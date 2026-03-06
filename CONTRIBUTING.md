# Contributing to KONASH

Clone the repository:

```bash
git clone https://github.com/konaequity/openkona.git
cd openkona
```

Install the dependencies:

```bash
uv sync --group dev
```

### Code Quality Checks (prek)

This project uses [prek](https://github.com/j178/prek) to run local checks (ruff, pyright, uv.lock sync, and unit tests). Before submitting a pull request, please ensure your code passes all quality checks:

```bash
# Install git hooks (optional but recommended)
uv run prek install

# Run all checks against all files (formatting, linting, typecheck, uv.lock, tests)
uv run prek run --all-files
```

You can also run individual hooks:

```bash
uv run prek run ruff
uv run prek run ruff-format
uv run prek run pyright
uv run prek run uv-lock-check
uv run prek run pytest
```

These checks are automatically run in CI for all pull requests.

### Release Process

To create a new release:

1. **Review merged PRs since the last release**:
   - Go to the [pull requests page](https://github.com/konaequity/openkona/pulls?q=is%3Apr+is%3Amerged+sort%3Aupdated-desc)
   - Review PRs merged since the last release to understand what changed
   - Note any breaking changes, new features, or important bug fixes

2. **Create a draft release**:
   - Go to [Actions](https://github.com/konaequity/openkona/actions)
   - Click "Run workflow"
   - Select the version bump type:
     - `patch`: Bug fixes and minor changes (0.1.0 → 0.1.1)
     - `minor`: New features and non-breaking changes (0.1.0 → 0.2.0)
     - `major`: Breaking changes (0.1.0 → 1.0.0)

3. **Edit the draft release notes**:
   - Go to the [releases page](https://github.com/konaequity/openkona/releases)
   - Click "Edit" on the draft release
   - Add release highlights, breaking changes, and curated changelog

4. **Finalize the release**:
   - Review and merge the automatically created release PR
   - This will automatically create the git tag, publish release notes, and build the package

### GPU Training (Local or Cloud VM)

Copy the `.env.example` file to `.env` and set the environment variables:

```bash
cp .env.example .env
```

Make sure you're on a machine with at least one A100-80GB or H100 GPU. Lower-end GPUs may work for smaller models with QLoRA, but training will be slower.

### Running an Example

Once set up, you can run one of the example notebooks:

```bash
cd examples/trivia_night
jupyter notebook trivia_night.ipynb
```

You can monitor training progress with Weights & Biases.

You should see improvement in `val/reward` after the first OAPL iteration completes.

### Adding Docs

We use Mintlify to serve our docs. Here are the steps for adding a new page:

1. Clone the repo
2. Open the `/docs` directory in your CLI and IDE
3. Run `npx mintlify dev` to start serving a local version of the docs in your browser
4. Create a new `.mdx` file in the relevant directory
5. Add a title and sidebar title (see other pages for examples)
6. In `docs.json`, add a link to the new page within one of the `navigation`.`groups`
7. Ensure everything works by navigating to and viewing the page in your browser
8. Submit a PR

### Cleaning Up

When you're done, shut down your GPU instance (if using a cloud VM) or stop the local training process.
