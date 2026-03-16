
# Automate_Research
Try to automate everything for research

# Interactive Tricks Browser
If you store many command snippets in markdown, use the interactive helper:

```bash
python3 interactive_tricks.py
```

Useful commands:
```bash
# List every parsed snippet
python3 interactive_tricks.py --list

# Search snippets without entering interactive mode
python3 interactive_tricks.py --search "submodule"

# Launch a simple interactive website
python3 interactive_tricks.py --web

# Add custom markdown sources (optional)
python3 interactive_tricks.py --source git.md --source os.md --source anaconda.md --source server.md

# Export static site for GitHub Pages
python3 interactive_tricks.py --export-static /path/to/tin-xai.github.io/tricks
```

GitHub Pages deploy (`https://tin-xai.github.io/tricks`):
1. Run `--export-static` into your `tin-xai.github.io/tricks` folder.
2. Commit and push in the `tin-xai.github.io` repo.
3. The site is static and does not need Python server on GitHub.

Website tips:
- Use file chips to filter by use case (`git.md`, `os.md`, `anaconda.md`, `dotfiles/...`).
- Use `/` to focus search, `j/k` to move between tricks.
- Click `Only This File` inside a trick to narrow down quickly.

# Author
Thanh Tin Nguyen
