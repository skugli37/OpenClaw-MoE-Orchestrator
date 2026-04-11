# Production Architecture

## Runtime Layers

1. `data_pipeline.py`
   Fetches real market data, validates schema, and writes normalized datasets.

2. `models.py`
   Defines the MoE autoencoder architectures used by the detectors.

3. `runtime.py`
   Adapts DeepSpeed configuration to the local machine, prepares optimizer groups for MoE, and manages distributed lifecycle.

4. `pipelines.py`
   Owns the production workflows:
   - single-asset detection
   - multi-asset detection
   - integrated orchestration
   - mission artifact generation

5. `news.py`
   Retrieves verifiable external context from Google News RSS.

6. `reports.py` and `visualization.py`
   Generate operator-facing outputs and charts.

## Production Outputs

Every production workflow writes to deterministic directories:

- `data/processed/`
- `artifacts/`
- `docs/`

Mission and integrated orchestrator runs also write JSON metadata with timestamps and git revision information for traceability.

## Design Constraints

- No synthetic market data in the production path
- No hardcoded home-directory paths
- Fail-fast validation for malformed datasets
- Packaged entry points instead of script-to-script subprocess chains
- Experimental code isolated under `experiments/`
