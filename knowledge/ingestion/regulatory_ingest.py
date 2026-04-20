"""Seed ingestion for UK energy regulatory corpus.

Covers: ECO4, BUS, SEG, Ofgem price cap, VAT, smart meter rollout,
PAS 2035, Warm Home Discount, Great British Insulation Scheme (GBIS).
No external files required — all content is curated seed text.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

_CHUNKS: list[dict] = [
    {
        "source_id": "regulatory.eco4.overview",
        "source_type": "regulatory",
        "jurisdiction": "england_wales_scotland",
        "publication_date": "2026-01-01",
        "url": "https://www.gov.uk/energy-company-obligation",
        "heading_path": ["ECO4", "Overview"],
        "chunk_index": 0,
        "scheme": "ECO4",
        "text": """## ECO4 — Energy Company Obligation 4 (Overview)

ECO4 (Energy Company Obligation 4) is a government scheme that obligates large UK energy
suppliers to fund free energy-efficiency improvements for eligible low-income and
fuel-poor households.

**Scheme period:** April 2022 – March 2026 (extended to 31 December 2026 as of 2025).

**Funded measures (non-exhaustive):**
- Loft insulation
- Cavity wall insulation
- External or internal wall insulation (solid walls)
- Air-source heat pump (ASHP) installation
- Boiler upgrades (first-time central heating)
- Solar PV (under certain pathways)
- Heating controls (smart thermostats, TRVs)

**Who qualifies:**
- Properties with EPC band D, E, F, or G *and* the occupant receives a qualifying benefit
  (Universal Credit, Pension Credit, Child Tax Credit, Housing Benefit, ESA, JSA, etc.)
- Properties with EPC band E, F, or G on the "Fabric First" pathway even without benefits

**How to apply:**
Contact your energy supplier directly or use the government's ECO4 eligibility checker
at www.gov.uk/energy-company-obligation. Installers must be TrustMark-registered and
follow PAS 2035:2023.
""",
    },
    {
        "source_id": "regulatory.eco4.eligibility",
        "source_type": "regulatory",
        "jurisdiction": "england_wales_scotland",
        "publication_date": "2026-01-01",
        "url": "https://www.gov.uk/energy-company-obligation",
        "heading_path": ["ECO4", "Eligibility", "Qualifying Benefits"],
        "chunk_index": 1,
        "scheme": "ECO4",
        "text": """## ECO4 — Qualifying Benefits & EPC Requirements

To qualify for ECO4, a household must meet BOTH an EPC condition AND an income/benefit condition.

### EPC condition
- **EPC band D, E, F, or G** for most pathways
- EPC band E, F, or G for the Fabric First pathway (no benefit needed but landlord consent required for rented)

### Qualifying benefits (any one of):
- Universal Credit (UC)
- Pension Credit (Guarantee Credit element)
- Child Tax Credit (CTC) or Working Tax Credit (WTC) with income below £31,000
- Income-based Jobseeker's Allowance (JSA)
- Income-related Employment and Support Allowance (ESA)
- Income Support
- Housing Benefit
- Child Benefit (with income below £31,000 for larger households)

### Additional LA Flex (Local Authority Flexible Eligibility):
Local authorities may refer households not on the above benefits but who are
fuel-poor or on low incomes, as defined by the LA.

### Rented properties:
Landlords must consent. Under Minimum Energy Efficiency Standards (MEES),
landlords are required to reach EPC E by law; ECO4 can fund compliance.
""",
    },
    {
        "source_id": "regulatory.bus.overview",
        "source_type": "regulatory",
        "jurisdiction": "england_wales",
        "publication_date": "2026-01-01",
        "url": "https://www.gov.uk/apply-boiler-upgrade-scheme",
        "heading_path": ["Boiler Upgrade Scheme", "Overview"],
        "chunk_index": 0,
        "scheme": "BUS",
        "text": """## Boiler Upgrade Scheme (BUS) — Overview

The Boiler Upgrade Scheme (BUS) provides upfront capital grants to homeowners and
small landlords in England and Wales to replace fossil fuel boilers with low-carbon
heating systems.

**Current grant levels (from April 2024):**
- Air-source heat pump (ASHP): £7,500
- Ground-source heat pump (GSHP): £7,500
- Biomass boiler (off-gas-grid properties only): £5,000
- Heat battery (thermal storage): added to scheme in late 2025

**Scheme period:** April 2022 onwards. Funded through the UK Heat & Buildings Strategy.

**Who can apply:**
- Owner-occupiers in England or Wales
- Small landlords (properties up to 4 units)
- Properties must have an existing fossil fuel heating system (oil, gas, LPG, or electric)

**How it works:**
1. Get quotes from MCS-certified installers
2. Installer applies for the voucher on your behalf
3. Voucher is issued, valid for 3 months
4. Installer completes work and redeems the voucher directly

**Important:** The property must have a valid EPC with no outstanding recommendations
for loft or cavity wall insulation (or those measures must already have been completed).
""",
    },
    {
        "source_id": "regulatory.bus.eligibility",
        "source_type": "regulatory",
        "jurisdiction": "england_wales",
        "publication_date": "2026-01-01",
        "url": "https://www.gov.uk/apply-boiler-upgrade-scheme",
        "heading_path": ["Boiler Upgrade Scheme", "Eligibility"],
        "chunk_index": 1,
        "scheme": "BUS",
        "text": """## Boiler Upgrade Scheme (BUS) — Eligibility Criteria

### Property requirements:
- Located in England or Wales (Scotland has separate HEEPS/ASHP scheme)
- Existing building (not new-build receiving grant funding for heat pump)
- Current heating must be fossil-fuel based (gas boiler, oil boiler, LPG, storage heaters, or direct electric)
- Valid EPC issued in the last 10 years
- No outstanding EPC recommendations for loft or cavity wall insulation

### Applicant requirements:
- Owner-occupier OR small private landlord
- The BUS grant is assigned to the MCS-certified installer, not the homeowner
- One grant per property address

### Installer requirements:
- Must be MCS (Microgeneration Certification Scheme) certified
- Heat pump must use a refrigerant with GWP < 2000

### Combining BUS with ECO4:
BUS and ECO4 cannot be stacked on the same measure. However, ECO4 can fund
insulation improvements as a separate measure to prepare a home for a BUS heat pump.
""",
    },
    {
        "source_id": "regulatory.seg.overview",
        "source_type": "regulatory",
        "jurisdiction": "england_wales_scotland",
        "publication_date": "2026-01-01",
        "url": "https://www.ofgem.gov.uk/environmental-and-social-schemes/smart-export-guarantee-seg",
        "heading_path": ["Smart Export Guarantee", "Overview"],
        "chunk_index": 0,
        "scheme": "SEG",
        "text": """## Smart Export Guarantee (SEG) — Overview

The Smart Export Guarantee (SEG) is a mandatory scheme requiring licensed electricity
suppliers with 150,000+ customers to offer export tariffs to small-scale low-carbon
generators (solar PV, wind, micro-CHP, hydro, anaerobic digestion).

**Replaced the Feed-in Tariff (FiT)** which closed 31 March 2019. Existing FiT recipients
are unaffected and continue to receive payments.

**Export rates (indicative, check your supplier):**
- Rates vary by supplier and product
- Some suppliers offer fixed rates (e.g., 15–24p/kWh)
- Octopus Outgoing offers Agile-linked variable export rates tied to half-hourly prices
- Zero rates are prohibited — suppliers must offer above zero

**Eligible technologies:**
- Solar PV (≤5 MW)
- Wind (≤5 MW)
- Micro-CHP (≤50 kW)
- Hydro (≤5 MW)
- Anaerobic digestion (≤5 MW)

**Requirements:**
- Smart meter capable of half-hourly export metering (SMETS2 or SMETS1 with comms hub)
- MCS-certified installation (for solar PV)
- No requirement to claim generation tariff (unlike FiT)
""",
    },
    {
        "source_id": "regulatory.gbis.overview",
        "source_type": "regulatory",
        "jurisdiction": "england_wales_scotland",
        "publication_date": "2026-01-01",
        "url": "https://www.gov.uk/great-british-insulation-scheme",
        "heading_path": ["Great British Insulation Scheme", "Overview"],
        "chunk_index": 0,
        "scheme": "GBIS",
        "text": """## Great British Insulation Scheme (GBIS) — Overview

The Great British Insulation Scheme (GBIS, formerly ECO+) is a government scheme that
funds insulation for a broader range of homes than ECO4, including higher-band EPC
properties that would not qualify for ECO4.

**Scheme period:** April 2023 – March 2026.

**Funded measures:**
- Loft insulation
- Cavity wall insulation
- Solid wall insulation
- Underfloor insulation
- Park home insulation
- Room-in-roof insulation

**Two eligibility routes:**
1. **Group A (Income-based):** Properties in Council Tax bands A–D (England) or A–E (Scotland/Wales)
   *and* the occupant receives a qualifying benefit (same list as ECO4)
2. **Group B (EPC-based):** Properties with EPC band D or E with no benefit requirement —
   the property must be in the lower 50% of Council Tax bands

**Maximum grant:** Typically covers the full cost for Group A. Group B may require a
contribution from the homeowner or landlord.

**Combining with ECO4:** GBIS and ECO4 can both fund insulation but on different pathways.
Installers must be TrustMark-registered.
""",
    },
    {
        "source_id": "regulatory.whd.overview",
        "source_type": "regulatory",
        "jurisdiction": "england_wales_scotland",
        "publication_date": "2026-01-01",
        "url": "https://www.gov.uk/the-warm-home-discount-scheme",
        "heading_path": ["Warm Home Discount", "Overview"],
        "chunk_index": 0,
        "scheme": "WHD",
        "text": """## Warm Home Discount (WHD) — Overview

The Warm Home Discount is a one-off annual electricity bill rebate of £150 for
eligible low-income households in England, Scotland, and Wales.

**Who qualifies:**
- **Core group (automatic):** Households receiving the Guarantee Credit element of
  Pension Credit. Rebate applied automatically.
- **Broader group (application required):** Households in receipt of qualifying means-tested
  benefits AND living in a property with a high heat demand (based on EPC, property type,
  and heating system). In England/Scotland, automatic data matching since 2022.

**Payment:** A £150 credit applied directly to your electricity account, usually between
October and March.

**Suppliers:** Participation is mandatory for suppliers with 150,000+ domestic customers.

**Important notes:**
- WHD is separate from the Winter Fuel Payment (pensioners only) and Cold Weather Payment
- Does not affect entitlement to ECO4 or GBIS
""",
    },
    {
        "source_id": "regulatory.ofgem_price_cap.overview",
        "source_type": "regulatory",
        "jurisdiction": "england_wales_scotland",
        "publication_date": "2026-01-01",
        "url": "https://www.ofgem.gov.uk/check-if-energy-price-cap-affects-you",
        "heading_path": ["Ofgem Price Cap", "Overview"],
        "chunk_index": 0,
        "text": """## Ofgem Default Tariff (Price) Cap — Overview

The Ofgem Default Tariff Cap is a regulatory limit on the unit rates and standing charges
suppliers can charge customers on default (variable) tariffs.

**Key facts:**
- Reviewed quarterly (January, April, July, October)
- Caps the *rate*, not the total bill — high energy users still pay more
- Applies to electricity and gas default variable tariffs
- Does NOT apply to fixed-term contracts

**Current cap rates (Q1 2026, typical dual-fuel):**
- Electricity unit rate: approximately 24.5p/kWh
- Gas unit rate: approximately 6.3p/kWh
- Electricity standing charge: approximately 61p/day
- Gas standing charge: approximately 31p/day
(Use get_active_tariff() API for precise current rates — these figures are illustrative)

**History:** Introduced October 2018. In 2022–2023 the cap rose sharply due to
wholesale gas price shocks following the Russia–Ukraine conflict.

**Prepayment meter cap:** A separate cap applies to prepayment meters (PPM). Rates have
been equalised with direct debit customers since October 2022.
""",
    },
    {
        "source_id": "regulatory.vat_energy.overview",
        "source_type": "regulatory",
        "jurisdiction": "uk",
        "publication_date": "2026-01-01",
        "url": "https://www.gov.uk/guidance/energy-products-reduced-rate-of-vat",
        "heading_path": ["VAT on Energy", "Overview"],
        "chunk_index": 0,
        "text": """## VAT on Domestic Energy

Domestic energy (electricity, gas, and other heating fuels) is subject to a reduced rate
of **5% VAT** in the UK — not the standard 20% rate.

**5% reduced rate applies to:**
- Electricity and gas for domestic use
- Oil and solid fuels for domestic use
- Small businesses consuming fewer than 1,000 kWh of electricity per month
  or 4,397 kWh of gas per month

**Standard 20% rate applies to:**
- Commercial energy supplies above the small-business threshold
- Installation services (e.g., heat pump installation) — though some energy-saving
  materials attract 0% VAT since April 2022

**Zero-rated (0% VAT) energy-saving materials (from April 2022):**
- Solar PV panels
- Air-source and ground-source heat pumps
- Wind turbines
- Water turbines
- Insulation for walls, floors, ceilings
- Draught stripping
- Central heating and hot water controls (including smart thermostats)

**Bills:** Energy bills already include VAT at 5%. The unit rate shown on your bill
is inclusive of VAT.
""",
    },
    {
        "source_id": "regulatory.mees.overview",
        "source_type": "regulatory",
        "jurisdiction": "england_wales",
        "publication_date": "2026-01-01",
        "url": "https://www.gov.uk/guidance/domestic-private-rented-property-minimum-energy-efficiency-standard-landlord-guidance",
        "heading_path": ["MEES", "Overview"],
        "chunk_index": 0,
        "text": """## Minimum Energy Efficiency Standards (MEES) for Private Rented Properties

MEES sets minimum energy performance requirements for private rented properties in
England and Wales.

**Current minimum (since April 2020):**
- EPC band E or above required for all new and existing private tenancies
- Landlords cannot let a property below EPC E (subject to exemptions)
- Applies to England and Wales only (Scotland has its own Private Residential Tenancy rules)

**Proposed future standards (subject to consultation):**
- EPC band C proposed as minimum for new tenancies by 2025 and all tenancies by 2028
  (timeline subject to government policy updates)

**Exemptions:**
- Maximum cost cap: landlords are exempt if improvement costs exceed the cap (£3,500 per property)
- All improvements made: if all relevant improvements have been made but EPC remains below E
- Devaluation exemption: written evidence that improvements would reduce property value by >5%
- Temporary exemption: 6-month exemption for new landlords

**Link to ECO4:** MEES compliance is a driver for ECO4 and GBIS applications for landlords.
Properties failing MEES cannot be legally let; ECO4 can fund required improvements.
""",
    },
    {
        "source_id": "regulatory.pas2035.overview",
        "source_type": "regulatory",
        "jurisdiction": "uk",
        "publication_date": "2023-01-01",
        "url": "https://www.bsigroup.com/en-GB/insights-and-media/insights/brochures/pas-2035-2030/",
        "heading_path": ["PAS 2035", "Overview"],
        "chunk_index": 0,
        "text": """## PAS 2035:2023 — Retrofitting Dwellings for Improved Energy Efficiency

PAS 2035:2023 is the publicly available specification for whole-house retrofit of
domestic buildings in the UK. All ECO4 and GBIS work must comply with PAS 2035.

**Key roles defined:**
- **Retrofit Assessor:** Surveys the property and produces a Whole House Plan
- **Retrofit Coordinator:** Manages the project, specifies measures, ensures compliance
- **Retrofit Installer:** Carries out the physical work; must be TrustMark-registered

**Whole House Plan (WHP):**
A holistic improvement roadmap for the property. Must be produced before any ECO4 or
GBIS measures are installed. Considers interactions between measures (e.g., adding
insulation before upgrading ventilation to avoid moisture issues).

**Significance:**
- Prevents substandard retrofits (e.g., poorly installed insulation causing damp)
- Required for ECO4, GBIS, and most LA schemes
- Introduced after widespread failures with solid wall insulation pre-2018

**TrustMark:**
All retrofit work under ECO4/GBIS must be carried out by TrustMark-registered businesses.
TrustMark is a government-endorsed quality scheme.
""",
    },
    {
        "source_id": "regulatory.smart_meters.overview",
        "source_type": "regulatory",
        "jurisdiction": "uk",
        "publication_date": "2026-01-01",
        "url": "https://www.gov.uk/guidance/smart-meters-how-they-work",
        "heading_path": ["Smart Meters", "Overview"],
        "chunk_index": 0,
        "text": """## Smart Meters in the UK — Overview

The UK smart meter rollout aims to replace all traditional meters with smart meters (SMETS2).

**SMETS1 vs SMETS2:**
- SMETS1: First-generation meters. May lose smart functionality if you switch supplier
  (known as "going dumb"). Over 12 million installed.
- SMETS2: Second-generation. Communicate via the national Data Communications Company (DCC)
  network and remain fully functional across supplier switches.

**Data granularity:**
- Electricity: half-hourly (HH) readings by default; can be set to 1-minute or daily reads
- Gas: daily reads by default; half-hourly technically possible

**Consumer Data Access:**
- Consumers can request HH data from their supplier
- Some suppliers provide API access (e.g., Octopus Energy n3rgy)
- Home Area Network (HAN) — SMETS2 meters expose a Zigbee interface for IHDs and
  consumer access devices (CADs)

**Obligations:**
- Suppliers are obligated to install smart meters on request — free of charge
- Annual smart meter rollout targets set by BEIS/DESNZ
""",
    },
]


def generate_seed_chunks() -> Iterator[dict]:
    yield from _CHUNKS


def write_chunks(chunks: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        meta = {k: v for k, v in chunk.items() if k != "text"}
        text = chunk["text"]
        safe_id = chunk["source_id"].replace(".", "_").replace("/", "_")
        out_path = out_dir / f"{safe_id}.md"
        out_path.write_text(
            f"<!--\n{json.dumps(meta, indent=2)}\n-->\n\n{text}",
            encoding="utf-8",
        )
