# GBD 2023 Dengue Manuscript

## Structure

- `manuscript.qmd` - Main manuscript text
- `figures/` - Generated figures for manuscript
- `tables/` - Generated tables for manuscript
- `data/` - Data files used in manuscript
- `references.bib` - Bibliography (exported from Zotero)
- `lancet.csl` - Citation style (Lancet format)

## Development Notebooks

Development notebooks are located in `notebooks/GBD2023_DENV/` (repository root):
- `01_data_loading.qmd` - Data loading and validation
- `02_figures.qmd` - Figure generation
- `03_tables.qmd` - Table generation
- `04_text_statistics.qmd` - In-text statistics

## Workflow

1. Develop analysis in notebooks (in `notebooks/GBD2023_DENV/`)
2. Notebooks save outputs to manuscript folder
3. Reference figures/tables in `manuscript.qmd`
4. When ready to finalize, move notebooks into manuscript folder

## Preview Manuscript

```bash
cd src/idd_manuscripts/GBD2023_DENV
quarto preview manuscript.qmd
```

## Render to Word

```bash
quarto render manuscript.qmd --to docx
```
