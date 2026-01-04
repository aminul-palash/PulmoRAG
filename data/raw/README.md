# COPD Document Collection

Place your collected COPD-related documents in the appropriate subdirectories:

## Directory Structure

### `guidelines/`
**Clinical practice guidelines and protocols**
- GOLD (Global Initiative for Chronic Obstructive Lung Disease) guidelines
- ATS (American Thoracic Society) guidelines
- ERS (European Respiratory Society) guidelines
- National/regional COPD management protocols

**Example sources:**
- https://goldcopd.org/
- https://www.thoracic.org/
- https://www.ersnet.org/

### `research/`
**Research papers and clinical studies**
- Clinical trials on COPD treatments
- Meta-analyses and systematic reviews
- Pharmacological studies
- Non-pharmacological intervention studies

**Example sources:**
- PubMed/MEDLINE
- Google Scholar
- Medical journals (NEJM, Lancet, etc.)

### `patient_education/`
**Patient information and education materials**
- COPD overview and symptoms
- Treatment explanations
- Lifestyle management guides
- Self-care instructions

**Example sources:**
- WHO materials
- CDC patient resources
- Patient advocacy organizations
- Hospital patient education materials

## Supported File Formats

- **PDF** (.pdf)
- **Word Documents** (.docx, .doc)
- **Text Files** (.txt, .md)

## Document Collection Tips

1. **Prioritize authoritative sources**: Official guidelines, peer-reviewed papers
2. **Include diverse perspectives**: Different treatment approaches, recent research
3. **Update regularly**: Medical knowledge evolves rapidly
4. **Check licenses**: Ensure you have rights to use the documents
5. **Organize by category**: Keep directory structure clean

## Quick Start

```bash
# Download sample GOLD guideline (example)
cd data/raw/guidelines/
wget https://goldcopd.org/wp-content/uploads/2023/03/GOLD-2023-ver-1.3-17Feb2023_WMV.pdf

# Or manually place files in respective directories
```

## After Adding Documents

Run the data setup script to process documents:
```bash
python scripts/setup_data.py
```

This will:
1. Load all documents from raw directories
2. Extract and clean text
3. Chunk documents for optimal retrieval
4. Save processed chunks to `data/processed/`
