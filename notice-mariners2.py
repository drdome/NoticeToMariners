#!/usr/bin/env python3
"""
Fixed and hardened version of the LNM table-first parser.

What I changed (high-level):
- Made network checks in resolve_latest_available_lnm more tolerant (HEAD, fallback to GET).
- Hardened table extraction to handle pdfplumber Table objects or raw table lists and to avoid crashes when extract() fails.
- Improved header/column detection fallbacks (DESCRIPTION, DETAILS, STATUS, NAME; SUBCATEGORY, CATEGORY, TYPE).
- Defensive checks for missing files, empty tables, and unexpected table shapes.
- Minor type hint cleanups to be more widely compatible.
- Preserved original behaviour and CLI usage at the bottom.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pdfplumber
import pandas as pd
import requests

def build_lnm_url(
    target_date: Optional[date] = None,
    district_code: str = "01",
    base_url: str = "https://www.navcen.uscg.gov/sites/default/files/pdf/lnms",
) -> str:
    """
    Build the weekly LNM PDF URL from a date.

    Notes on filename format observed on NAVCEN:
      LNM{district}{week:02d}{year}.pdf
    Example: district 01, ISO week 44 of 2025 -> LNM01442025.pdf
    Using ISO week keeps the year aligned with USCG numbering around New Year's.
    """
    d = target_date or date.today()
    iso_year, iso_week, _ = d.isocalendar()
    filename = f"LNM{district_code}{iso_week:02d}{iso_year}.pdf"
    return f"{base_url}/{filename}"

def download_file(url: str, dest_path: str | Path, chunk_size: int = 1_048_576) -> Path:
    """Download a file with basic error handling; returns the destination path."""
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return dest_path

def normalize_targets(targets: Iterable[str]) -> List[str]:
    return [t.lower() for t in targets]

def resolve_latest_available_lnm(
    start_date: date,
    district_code: str,
    lookback_weeks: int = 6,
    base_url: str = "https://www.navcen.uscg.gov/sites/default/files/pdf/lnms",
) -> Optional[Tuple[str, date]]:
    """
    Walk backwards by week from start_date until an available PDF is found.
    Returns (url, date_used) or None if nothing found.
    """
    for weeks_back in range(0, lookback_weeks + 1):
        candidate_date = start_date - timedelta(weeks=weeks_back)
        url = build_lnm_url(candidate_date, district_code=district_code, base_url=base_url)
        try:
            # Prefer HEAD but some servers don't respond well to HEAD; try HEAD then GET if needed.
            resp = requests.head(url, allow_redirects=True, timeout=15)
            status = resp.status_code
            content_type = resp.headers.get("Content-Type", "")
            if status == 200 and content_type and "pdf" in content_type.lower():
                return url, candidate_date

            # If HEAD didn't confirm a PDF, try a lightweight GET (stream but no download)
            resp = requests.get(url, stream=True, timeout=15)
            status = resp.status_code
            content_type = resp.headers.get("Content-Type", "")
            if status == 200 and content_type and "pdf" in content_type.lower():
                return url, candidate_date

            # Some servers don't set Content-Type properly; accept 200 as a best-effort
            if status == 200:
                return url, candidate_date

        except requests.RequestException:
            # network blip or not found; continue searching previous weeks
            continue
    return None

def _safe_extract_table_rows(table_obj) -> Optional[List[List[str]]]:
    """
    Accept either a pdfplumber Table object (with .extract()) or a pre-extracted table
    (list-of-lists). Return None on failure.
    """
    if table_obj is None:
        return None
    # If it's already a list-of-lists (pdfplumber.page.extract_tables() style)
    if isinstance(table_obj, list):
        return table_obj
    # Otherwise try extract() method (pdfplumber Table)
    extract = getattr(table_obj, "extract", None)
    if callable(extract):
        try:
            rows = extract()
            return rows
        except Exception:
            return None
    return None

def parse_lnm_detailed(
    pdf_path: str,
    target_locations: Optional[Iterable[str]] = None,
    target_subcategories: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Table-first parser modeled after original logic with optional filters.
    target_locations/subcategories are case-insensitive; matches capture rows if either condition passes.
    """
    target_locations = list(target_locations or [])
    target_locations_norm = normalize_targets(target_locations) if target_locations else []
    target_subcategories = [s.lower() for s in (target_subcategories or [])]

    extracted_data = []

    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return pd.DataFrame()

    print(f"Opening PDF: {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Scanning {total_pages} pages...")

        for page_num, page in enumerate(pdf.pages):
            print(f"  Page {page_num + 1}/{total_pages}")
            # page.find_tables() returns Table objects; page.extract_tables() returns raw lists.
            tables = []
            try:
                tables = page.find_tables() or []
            except Exception:
                # Fallback to extract_tables if find_tables fails for some reason
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    tables = []

            page_hits = 0

            for table in tables:
                # Attempt to get bbox if available for header context searching
                bbox = getattr(table, "bbox", None)
                if bbox and len(bbox) >= 4:
                    x0, top, x1, bottom = bbox
                    header_search_area = (0, max(0, top - 120), page.width, top)
                else:
                    # No bbox available; search in top part of page
                    header_search_area = (0, 0, page.width, min(200, page.height / 3))

                try:
                    text_above = (
                        page.within_bbox(header_search_area).extract_text() or ""
                    )
                except Exception:
                    text_above = ""

                section_header = "Unknown"
                if text_above:
                    lines = [line.strip() for line in text_above.split("\n") if line.strip()]
                    if lines:
                        section_header = lines[-1]

                # Extract table data rows as list-of-lists
                data = _safe_extract_table_rows(table)
                if not data:
                    continue

                # If header row is empty (some tables have spacing rows), find first non-empty row to be headers
                header_row = None
                for r in data:
                    if any(cell not in (None, "", "\n") for cell in r):
                        header_row = r
                        break
                if header_row is None:
                    continue

                # Normalize headers to uppercase strings for matching
                headers = [str(cell).strip().upper() if cell is not None else "" for cell in header_row]

                # Identify Column Indices with fallbacks
                subcat_idx = -1
                for candidate in ("SUBCATEGORY", "CATEGORY", "TYPE"):
                    if candidate in headers:
                        subcat_idx = headers.index(candidate)
                        break

                desc_idx = -1
                for candidate in ("DESCRIPTION", "DETAILS", "TEXT", "STATUS", "NAME"):
                    if candidate in headers:
                        desc_idx = headers.index(candidate)
                        break

                # If the header row isn't actually a header but a data row (many LNM tables don't have headers),
                # we will treat the first row as header and then search for likely columns by name; if none found,
                # we'll fallback to reasonable defaults later.
                header_row_idx = data.index(header_row)

                # Iterate rows after header_row_idx
                for row in data[header_row_idx + 1 :]:
                    # Clean row data for text search
                    clean_row = [str(cell).replace("\n", " ").strip() if cell is not None else "" for cell in row]
                    row_text = " ".join(clean_row)
                    row_lower = row_text.lower()

                    # Determine subcategory value
                    subcategory_val = "General"
                    if subcat_idx != -1 and len(clean_row) > subcat_idx and clean_row[subcat_idx]:
                        subcategory_val = clean_row[subcat_idx]
                    else:
                        # Heuristic: if headers contain LLNR or AID TYPE, label as Discrepancy
                        if any(h in ("LLNR", "AID TYPE") for h in headers):
                            subcategory_val = "Discrepancy"

                    subcat_lower = str(subcategory_val).lower()

                    # Location matching: check section header and row text
                    section_lower = section_header.lower()
                    section_location_match = bool(target_locations_norm) and any(
                        loc in section_lower for loc in target_locations_norm
                    )
                    location_row_match = bool(target_locations_norm) and any(
                        loc in row_lower for loc in target_locations_norm
                    )
                    location_hit = section_location_match or location_row_match

                    # Subcategory matching
                    subcat_hit = bool(target_subcategories) and (
                        any(s in subcat_lower for s in target_subcategories)
                        or any(s in row_lower for s in target_subcategories)
                        or any(s in section_lower for s in target_subcategories)
                    )

                    should_capture = (
                        (bool(target_locations_norm) and location_hit)
                        or (bool(target_subcategories) and subcat_hit)
                        or (not target_locations_norm and not target_subcategories)
                    )

                    if should_capture:
                        # Capture description
                        description_val = ""
                        if desc_idx != -1 and len(clean_row) > desc_idx and clean_row[desc_idx]:
                            description_val = clean_row[desc_idx]
                        else:
                            # Try to use columns that look like description: pick the longest non-empty cell
                            non_empty_cells = [c for c in clean_row if c]
                            if non_empty_cells:
                                description_val = max(non_empty_cells, key=len)
                            else:
                                description_val = " | ".join(clean_row)

                        extracted_data.append(
                            {
                                "Page": page_num + 1,
                                "Section Context": section_header,
                                "Subcategory": subcategory_val,
                                "Description": description_val,
                                "Full Data": " | ".join(clean_row),
                            }
                        )
                        page_hits += 1

            if page_hits:
                print(f"    Found {page_hits} hits on page {page_num + 1}")

    if not extracted_data:
        return pd.DataFrame()

    return pd.DataFrame(extracted_data)


if __name__ == "__main__":
    # --- CONFIGURATION ---
    target_locations = ["Long Island Sound", "LIS", "New York", "Connecticut", "New Haven", "New London"]
    target_subcategories = ["Marine Events", "Hazards to Navigation"]
    district = "01"  # Northeast
    anchor_date = None  # e.g., date(2025, 10, 29)
    anchor_env = os.getenv("LNM_ANCHOR_DATE")
    if anchor_env:
        try:
            anchor_date = date.fromisoformat(anchor_env)
        except ValueError:
            print(f"Could not parse LNM_ANCHOR_DATE={anchor_env}; falling back to today.")
    today = date.today()

    # --- FIND TARGET PDF ---
    if anchor_date:
        pdf_url = build_lnm_url(anchor_date, district_code=district)
        used_date = anchor_date
        print(f"Using anchor date {anchor_date} -> {pdf_url}")
    else:
        resolved = resolve_latest_available_lnm(today, district_code=district)
        if not resolved:
            print(f"No available LNM found for district {district} within the last few weeks starting {today}")
            raise SystemExit(1)
        pdf_url, used_date = resolved
        # used_date is a date object
        print(f"Using week {used_date.isocalendar()[1]} {used_date.isocalendar()[0]} -> {pdf_url}")

    # --- DOWNLOAD ---
    download_to = Path.home() / "Downloads" / Path(pdf_url).name
    print(f"Downloading {pdf_url} -> {download_to}")
    try:
        downloaded_pdf = download_file(pdf_url, download_to)
    except Exception as e:
        print(f"Download failed: {e}")
        raise SystemExit(1)

    # --- RUN ---
    df = parse_lnm_detailed(
        str(downloaded_pdf),
        target_locations=target_locations,
        target_subcategories=target_subcategories,
    )

    # --- OUTPUT ---
    timestamp = used_date.strftime("%Y%m%d") if used_date else date.today().strftime("%Y%m%d")
    output_basename = f"LIS_Weekly_Report_Detailed_{timestamp}.csv"
    dump_basename = f"LNM_full_dump_{timestamp}.csv"

    if not df.empty:
        print(f"\nSuccess! Found {len(df)} relevant entries.\n")

        # Display preview
        pd.set_option("display.max_colwidth", 60)
        print(df[["Section Context", "Subcategory", "Description"].head(10)])

        # Export
        output_csv = Path(download_to).with_name(output_basename)
        df.to_csv(output_csv, index=False)
        print(f"\nReport saved to: {output_csv}")
    else:
        print("No matches found with current filters. Dumping full tables for inspection...")
        full_df = parse_lnm_detailed(str(downloaded_pdf), target_locations=[], target_subcategories=[])
        dump_csv = Path(download_to).with_name(dump_basename)
        full_df.to_csv(dump_csv, index=False)
        print(f"Full table dump saved to: {dump_csv}")
