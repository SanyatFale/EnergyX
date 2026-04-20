"""Seed ingestion for energy efficiency guidance corpus.

Covers: insulation types, heat pumps, solar PV, LED lighting, appliance efficiency,
heating controls, draught-proofing, hot water, EV charging optimisation.
No external files required — all content is curated seed text.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

_CHUNKS: list[dict] = [
    {
        "source_id": "efficiency.insulation.loft",
        "source_type": "efficiency_guide",
        "jurisdiction": "uk",
        "publication_date": "2026-01-01",
        "url": "https://energysavingtrust.org.uk/advice/roof-and-loft-insulation/",
        "heading_path": ["Insulation", "Loft Insulation"],
        "chunk_index": 0,
        "text": """## Loft Insulation

Insulating your loft is one of the most cost-effective ways to reduce heat loss.
Around 25% of heat in an uninsulated home is lost through the roof.

**Recommended depth:** 270 mm of mineral wool (glass or rock wool) or equivalent.

**Types:**
- **Cold loft (flat ceiling):** Lay insulation between and over the joists on the loft floor.
  Do not insulate the roof slope — this keeps the loft cold and prevents condensation.
- **Warm loft (converted):** Insulate the rafters (between and below), using rigid foam boards
  or spray foam with adequate ventilation gaps.

**Cost (indicative):** £300–£600 for a typical detached house, DIY materials.
Professional installation typically £400–£1,200 depending on size and access.

**Savings (indicative):** Up to £150–£250/year for a semi-detached house depending on
existing insulation depth and current fuel prices (use predict_bill() for your home).

**ECO4/GBIS:** Loft insulation is one of the primary measures funded under both schemes.

**Caution:** Ensure existing insulation is dry and free from pest damage before topping up.
"""
    },
    {
        "source_id": "efficiency.insulation.cavity_wall",
        "source_type": "efficiency_guide",
        "jurisdiction": "uk",
        "publication_date": "2026-01-01",
        "url": "https://energysavingtrust.org.uk/advice/cavity-wall-insulation/",
        "heading_path": ["Insulation", "Cavity Wall Insulation"],
        "chunk_index": 0,
        "text": """## Cavity Wall Insulation

Most UK homes built after 1930 have cavity walls — two layers of brick or block
with a gap (cavity) in between. Filling this cavity with insulating material reduces
heat loss through walls, which accounts for around 35% of heat lost in uninsulated homes.

**Types of fill:**
- **Mineral wool (blown):** Most common, reversible, suitable for most cavities
- **EPS beads:** Polystyrene beads with adhesive; good for narrower cavities
- **Polyurethane foam:** Not recommended by many architects; difficult to remove

**Suitability checks:**
- Cavity width must be ≥50 mm
- Wall must be in good condition (no damp, cracks, or pointing failures)
- Not suitable if exposed to driving rain (check postcode designation)
- Get a survey first — unsuitable injection can cause damp

**Savings (indicative):** £150–£300/year for a typical semi-detached house.

**Grants:** Funded under ECO4 and GBIS. Free for eligible households.

**Warning:** Poorly installed cavity wall insulation can introduce damp and reduce air quality.
Always use a TrustMark/CIGA-approved installer.
"""
    },
    {
        "source_id": "efficiency.insulation.solid_wall",
        "source_type": "efficiency_guide",
        "jurisdiction": "uk",
        "publication_date": "2026-01-01",
        "url": "https://energysavingtrust.org.uk/advice/solid-wall-insulation/",
        "heading_path": ["Insulation", "Solid Wall Insulation"],
        "chunk_index": 0,
        "text": """## Solid Wall Insulation

Homes built before 1920 (and many built before 1940) typically have solid walls with no
cavity. Solid walls lose about twice as much heat as cavity walls.

**External Wall Insulation (EWI):**
- Rigid foam or mineral wool boards fixed to the outside of the wall, finished with
  render, cladding, or brick slip
- Does not reduce internal floor area
- Improves thermal bridging and airtightness
- Typical cost: £8,000–£22,000 for a semi-detached house
- Typical U-value improvement: from ~2.1 W/m²K to ~0.3 W/m²K

**Internal Wall Insulation (IWI):**
- Rigid foam boards or stud-frame with insulation fitted internally
- Reduces room size slightly (typically 80–120 mm per wall)
- Requires careful treatment of junctions to avoid thermal bridging
- Typical cost: £5,000–£15,000

**Savings (indicative):** £200–£400/year for a semi-detached.

**Grants:** ECO4 funds solid wall insulation under the Fabric First pathway for EPC bands
E, F, G. GBIS can fund it for EPC band D.

**PAS 2035 requirement:** All solid wall insulation under ECO4/GBIS must follow
PAS 2035:2023 and be specified by a Retrofit Coordinator.
"""
    },
    {
        "source_id": "efficiency.heating.heat_pump_guide",
        "source_type": "efficiency_guide",
        "jurisdiction": "uk",
        "publication_date": "2026-01-01",
        "url": "https://energysavingtrust.org.uk/advice/heat-pumps/",
        "heading_path": ["Heating", "Heat Pumps"],
        "chunk_index": 0,
        "text": """## Heat Pumps — Practical Guide for UK Homes

A heat pump moves heat from outside air (ASHP) or the ground (GSHP) into your home.
It does not generate heat — it transfers it, achieving 2.5–4× efficiency vs direct electric.

**Air-Source Heat Pump (ASHP):**
- Extracts heat from outdoor air even down to –20°C
- Requires outside unit (similar size to an air-conditioning unit)
- Best installed alongside good insulation (target EPC B or C first)
- Typical seasonal CoP: 2.5–3.5
- Running cost depends heavily on electricity price and insulation level

**Ground-Source Heat Pump (GSHP):**
- Uses horizontal ground loops (requires garden area ≥ 200 m²) or vertical borehole
- More stable efficiency year-round (ground temperature ~10–12°C constant in UK)
- Typical seasonal CoP: 3.0–4.5
- Higher installation cost (£15,000–£35,000 vs £7,000–£15,000 for ASHP)

**Is my home suitable?**
- Well-insulated homes (EPC A/B/C) are best candidates
- Must have space for outdoor unit (ASHP) or ground loops/borehole (GSHP)
- Radiators may need upsizing for low flow temperatures (45–55°C vs 70–80°C for gas)
- Underfloor heating is ideal for heat pump operation

**BUS grant:** £7,500 from the Boiler Upgrade Scheme. See regulatory.bus.overview.

**Running costs:** Depend on current electricity price vs gas price ratio.
A CoP of 3 at 24p/kWh electricity = 8p/kWh equivalent heat cost.
Check get_active_tariff() for current rates.
"""
    },
    {
        "source_id": "efficiency.heating.thermostat_controls",
        "source_type": "efficiency_guide",
        "jurisdiction": "uk",
        "publication_date": "2026-01-01",
        "url": "https://energysavingtrust.org.uk/advice/home-heating-controls/",
        "heading_path": ["Heating", "Controls and Thermostats"],
        "chunk_index": 0,
        "text": """## Heating Controls — Thermostats, TRVs, and Programmers

Good heating controls let you heat your home only when needed and only in rooms that
are occupied, saving 10–15% on heating bills.

**Key controls:**
- **Programmer / timer:** Sets on/off times for central heating and hot water
- **Room thermostat:** Maintains a target temperature; cuts heating when reached.
  Recommended setting: 18–21°C (each degree lower saves ~10% on heating)
- **Thermostatic radiator valves (TRVs):** Control individual radiators, enabling
  room-by-room temperature zoning. Use TRV 1–2 in bedrooms, TRV 3–4 in living areas
- **Smart thermostat (e.g., Hive, Nest, tado°):** Remote control via app, geofencing,
  learning algorithms, integration with voice assistants

**Best practice:**
1. Set the programmer to your normal occupancy pattern (overnight setback of 15–16°C)
2. Use TRVs in all rooms except the one with the room thermostat
3. Set the room thermostat to your desired temperature — do NOT turn it to max to
   heat the room faster (it won't; the boiler heats at the same rate regardless)
4. Use a hot water cylinder thermostat set to 60°C to prevent Legionella

**ECO4:** Smart controls and TRVs can be funded as part of a broader ECO4 measure package.
Heating controls alone are generally not eligible as a standalone ECO4 measure.
"""
    },
    {
        "source_id": "efficiency.solar.pv_guide",
        "source_type": "efficiency_guide",
        "jurisdiction": "uk",
        "publication_date": "2026-01-01",
        "url": "https://energysavingtrust.org.uk/advice/solar-panels/",
        "heading_path": ["Solar", "Solar PV Guide"],
        "chunk_index": 0,
        "text": """## Solar PV — Guide for UK Homes

Solar photovoltaic (PV) panels convert sunlight into electricity. Even the UK's
cloudy climate provides sufficient irradiance for useful generation.

**Typical system size:** 3–4 kWp for a 3–4 bedroom house (10–13 panels, ~25 m²).

**Annual generation (indicative):**
- South-facing, 30° tilt: ~850–950 kWh/kWp/year
- East/West split: ~700–800 kWh/kWp/year
- Reduces your import bill by displacing grid electricity (self-consumption)

**Self-consumption vs export:**
- Electricity used in the home saves ~24p/kWh (current cap rate)
- Excess exported earns ~15–24p/kWh via the Smart Export Guarantee (SEG)
- A battery storage system (e.g., Tesla Powerwall, GivEnergy) increases self-consumption
  to 70–80% (from ~25–35% without battery)

**Costs (indicative):**
- 3.5 kWp system: £6,000–£9,000 installed
- Battery (10 kWh): additional £4,000–£7,000
- Simple payback: 7–12 years depending on export rate and self-consumption

**VAT:** Zero-rated for residential solar PV since April 2022.

**EPC impact:** Solar PV typically improves EPC by 1–2 bands.

**SEG:** Register with your supplier to get paid for exports. See regulatory.seg.overview.
"""
    },
    {
        "source_id": "efficiency.hot_water.cylinder",
        "source_type": "efficiency_guide",
        "jurisdiction": "uk",
        "publication_date": "2026-01-01",
        "url": "https://energysavingtrust.org.uk/advice/water-heating/",
        "heading_path": ["Hot Water", "Cylinder and Immersion Heater"],
        "chunk_index": 0,
        "text": """## Hot Water — Cylinders, Immersion Heaters, and Heat Pumps

**Hot water cylinder:**
- A well-insulated cylinder (factory-insulated or with a 75 mm jacket) retains heat
  for 4–8 hours with minimal loss
- Set cylinder thermostat to 60°C for Legionella prevention; weekly heat to 60°C if
  using a heat pump (which normally operates at 45–55°C)
- A 180-litre cylinder typically meets the needs of a 3–4 person household

**Immersion heater:**
- 3 kW electric element; inefficient as primary heating (~24p/kWh at cap rate)
- Useful as backup or with off-peak tariff (Economy 7 or Octopus Go)
- With solar PV and a diverter (e.g., eddi, iBoost): use excess generation to heat water
  before exporting — this typically saves more than SEG export rates

**Hot water heat pump (HWWHP / heat pump cylinder):**
- COP of 2.5–3.5 for hot water only
- Cheaper to run than immersion at full rate; may not suit all households
- Grant available under BUS in some configurations

**Savings tips:**
1. Fix dripping hot taps — a dripping tap wastes up to 100 litres/day
2. Use a shower timer — 4 minutes vs 8 minutes halves hot water use
3. Set cylinder timer to match occupancy (not 24/7)
4. Insulate the first metre of pipe from the cylinder to reduce standby losses
"""
    },
    {
        "source_id": "efficiency.appliances.overview",
        "source_type": "efficiency_guide",
        "jurisdiction": "uk",
        "publication_date": "2026-01-01",
        "url": "https://energysavingtrust.org.uk/advice/home-appliances/",
        "heading_path": ["Appliances", "Energy Labels and Tips"],
        "chunk_index": 0,
        "text": """## Home Appliances — Energy Labels and Efficiency Tips

**EU/UK Energy Label (rescaled 2021):**
- Scale: A (most efficient) to G
- Applies to: washing machines, dishwashers, tumble dryers, ovens, TVs, refrigerators
- Replacing a G-rated washing machine with an A-rated model saves ~£40/year

**Typical appliance consumption (indicative):**
| Appliance | Typical W | Usage pattern | kWh/year |
|---|---|---|---|
| Fridge-freezer (A) | 30–100 W avg | 24/7 | 200–400 |
| Washing machine (A) | 700–2,000 W | 5 cycles/week | 100–200 |
| Tumble dryer (heat pump) | 500–900 W | 3 cycles/week | 100–200 |
| Dishwasher (A) | 1,200–2,100 W | 7 cycles/week | 200–300 |
| Electric shower (9.5 kW) | 9,500 W | 2× daily, 5 min | 350 |
| Kettle | 2,000–3,000 W | 4× daily, 2 min | 100 |
| LED bulb | 8–12 W | 4 hours/day | 10–15 |

**Energy-saving tips:**
1. Wash at 30°C and reduce cycles — modern detergents work well at low temperatures
2. Always run dishwasher and washing machine with full loads
3. Use the eco programme (typically 50–70% energy saving vs normal cycle)
4. Avoid standby — TVs and set-top boxes in standby can cost £20–£40/year
5. Defrost fridge-freezer regularly — ice build-up increases energy consumption
"""
    },
    {
        "source_id": "efficiency.draught_proofing.overview",
        "source_type": "efficiency_guide",
        "jurisdiction": "uk",
        "publication_date": "2026-01-01",
        "url": "https://energysavingtrust.org.uk/advice/draught-proofing/",
        "heading_path": ["Draught-Proofing", "Overview"],
        "chunk_index": 0,
        "text": """## Draught-Proofing

Draught-proofing is one of the cheapest, most cost-effective ways to reduce heating bills.
Uncontrolled draughts through gaps in doors, windows, floors, and chimneys account for
15–25% of heat loss in older UK homes.

**Common draught sources:**
- Letterboxes, keyholes, and external door gaps
- Window frames and sash windows
- Suspended timber floor boards
- Unused fireplaces and chimneys
- Loft hatches
- Pipes and cables passing through external walls

**DIY solutions (low cost):**
- Self-adhesive foam strip or brush strip for door and window frames (£5–£20)
- Door draught excluder (bottom seal)
- Chimney balloon (£20–£25) for unused fireplaces — must be removed before lighting a fire
- Expanding foam for pipe gaps (take care near electrical cables)

**Professional solutions:**
- Sash window refurbishment with brush seals (£200–£400 per window)
- Suspended floor insulation with floor board sealing

**Savings (indicative):** £60–£200/year depending on house age and size.

**Note:** Draught-proofing reduces uncontrolled infiltration but controllable ventilation
(e.g., trickle vents, extractor fans) is essential for air quality and moisture control.
Do not block background ventilation in rooms with gas appliances.
"""
    },
    {
        "source_id": "efficiency.ev_charging.optimisation",
        "source_type": "efficiency_guide",
        "jurisdiction": "uk",
        "publication_date": "2026-01-01",
        "url": "https://energysavingtrust.org.uk/advice/electric-vehicles/",
        "heading_path": ["Electric Vehicles", "Home Charging Optimisation"],
        "chunk_index": 0,
        "text": """## EV Home Charging — Optimisation for UK Households

**Home charging hardware:**
- Untethered 7.2 kW smart charger (e.g., Ohme, Zappi, Hypervolt) is the UK standard
- Charge Point Grant: £350 off a smart charger for flat/leasehold residents (as of 2026)
- 7.2 kW charges most EVs from 20% to 80% in ~5–6 hours overnight

**Optimal charging strategies:**
1. **Economy 7 / Octopus Go:** Schedule charging in the off-peak window (00:30–07:30).
   At ~7–13p/kWh off-peak vs 24p/kWh peak, savings are £300–£600/year for a typical EV.
2. **Intelligent Octopus Go / Agile Octopus:** Smart chargers communicate with the tariff
   to fill during the cheapest half-hourly slots automatically.
3. **Solar PV surplus:** Use a smart charger (Zappi, Ohme with solar integration) to
   divert excess solar generation to EV charging before exporting.
4. **Vehicle-to-Grid (V2G):** Emerging — discharge battery to home or grid at peak rates.
   Currently available with specific vehicle/charger combinations (e.g., Nissan Leaf + Indra).

**Carbon optimisation:**
- Schedule charging when grid carbon intensity is low (typically overnight in high-wind periods)
- Use the Carbon Intensity API (carbonintensity.org.uk) or EnergyX CarbonTracker for timing

**Demand Response:**
- Octopus Saving Sessions reward EV owners who pause charging during peak grid stress events
"""
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
