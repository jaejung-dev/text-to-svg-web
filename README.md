# Text-to-SVG Web Gallery

Standalone GitHub Pages site for reviewing production Text-to-SVG outputs
against the validation ground truth and three reference model outputs.

## Build Data

The source validation prompts and reference renders are read from
`/home/ubuntu/lica-score-web/report-data.json`. The source project is not
modified.

```bash
cd /home/ubuntu/text-to-svg-web
export BASETEN_API_KEY="..."
python scripts/build_site_data.py
```

For layout work without calling the model:

```bash
python scripts/build_site_data.py --skip-generation
```

## Preview

```bash
python -m http.server 8010
```

Open `http://localhost:8010`.

## Publish

This is a static site. Push this directory to a GitHub repository and enable
GitHub Pages from the repository root.
