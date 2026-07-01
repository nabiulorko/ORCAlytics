# ORCAlytics

A Streamlit app that parses ORCA `.out` files and computes Conceptual DFT
(Parr–Pearson) global reactivity descriptors from the HOMO/LUMO orbital
energies.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## What it does

1. Upload one or more ORCA `.out` files.
2. The app finds the final `ORBITAL ENERGIES` table, detects the HOMO
   (last orbital with non-zero occupation) and LUMO (first orbital with
   zero occupation), in eV.
3. It computes:
   - ΔE_gap = E_LUMO − E_HOMO
   - I = −E_HOMO, A = −E_LUMO
   - χ = (I + A)/2, μ = −χ
   - η = ΔE_gap/2, S = 1/η
   - ω = χ²/(2η)
4. Results are shown in a styled table, with CSV export.

## Files

- `app.py` — the full Streamlit application
- `requirements.txt` — Python dependencies
