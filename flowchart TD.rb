flowchart TD
    Q(["👤 User Query"]) --> UI["🖥️ Streamlit UI"]

    UI --> RT

    subgraph RT["ROUTER  classify_query()"]
        direction LR
        H["Keyword regex\nheuristics"] -->|no strong match| L["llama3.1-8b\nprompt: 'classify into\nknowledge / analysis /\ncontrol / status'"]
    end

    RT -->|knowledge| KA["📚 KnowledgeAgent\nanswer()"]
    RT -->|status| ST["📊 StatusAgent\nlive buffer read"]
    RT -->|control| CA["⚙️ ControlAgent\nhandle()"]
    RT -->|analysis| AA

    KA -->|"BM25 + ChromaDB → reranker\n→ LLM synthesizer"| R
    ST -->|"current readings · cost · carbon"| R
    CA -->|"permission check → HA JSON"| R

    subgraph AA["🔬 ANALYSIS AGENT"]
        direction TB

        subgraph PL["PHASE 1 — plan()"]
            PF["profile_dataset()\ncompute: mean · std · ADF test\nseasonality · autocorrelation"]
            PM["llama3.1-8b\nprompt: system role + dataset stats\n→ structured output UserTaskPlan\n{ task_type, models, horizon,\n  needs_explanation, is_multivariate }"]
            PF --> PM
        end

        PM --> SC{"single-tool\nshort-circuit?"}
        SC -->|yes| DI["direct tool call\n no LLM loop"]
        SC -->|no| PH2

        subgraph PH2["PHASE 2 — execute()  LLM tool-calling loop  max 25 steps"]
            direction LR
            LM["llama3.1-8b\nbind_tools()\nsystem: EnergyX expert\n'call ONE tool per response'"]
            TC["tool call\ndecision"]
            TR["tool result\n→ ToolMessage appended"]
            LM --> TC --> TR -->|next step| LM
        end

        DI --> TL
        PH2 --> TL

        subgraph TL["TOOLS  —  14 available"]
            direction LR
            T1["forecast_univariate\nforecast_multivariate"]
            T2["explain_forecast\nSHAP · FI · STL"]
            T3["detect_anomalies\n7-method ensemble"]
            T4["explain_anomalies"]
            T5["counterfactual_fwd\ncounterfactual_inv"]
            T6["get_cost_analysis\nmake_budget\nevaluate_tariff_switch"]
            T7["weather_impact_analysis"]
            T8["generate_report"]
        end

        subgraph FP["FORECAST PIPELINE  called by forecast tools"]
            direction TB
            FC["Feature Cache  data/feature_cache/*.parquet\nsparse lags 1–1440 min + Fourier sin/cos 24 h & 168 h\n15 features  ·  content-keyed MD5  ·  ~36 ms read"]
            CV["3-fold rolling-origin CV\nno JSON round-trip  ·  in-memory numpy"]
            MD["Models\nNaive · SeasonalNaive · ARIMA · ETS\nRandomForest · LightGBM · N-BEATS"]
            EN["Ensemble\ninverse-MAPE weighting"]
            FC --> CV --> MD --> EN
        end

        T1 --> FP

        subgraph DA["DATA  loaded once per session"]
            D1["data/ideal_hierarchy/{home}/*.parquet\n14-day window  ·  1-min resolution\nelectricity_apparent + appliance cols + weather"]
        end

        DA -.->|"y  X  df"| PL
        DA -.->|"y  X  df"| FP
    end

    TL --> R(["📤 Response"])
    R --> Q2(["👤 User"])

    classDef agent  fill:#1e3a5f,stroke:#3b82f6,color:#bfdbfe
    classDef llm    fill:#2d1b69,stroke:#7c3aed,color:#ede9fe
    classDef tool   fill:#14532d,stroke:#22c55e,color:#dcfce7
    classDef data   fill:#1c1917,stroke:#78716c,color:#d6d3d1
    classDef resp   fill:#0c4a6e,stroke:#38bdf8,color:#e0f2fe
    classDef entry  fill:#1e293b,stroke:#64748b,color:#f1f5f9

    class KA,ST,CA,AA agent
    class PM,LM,L llm
    class T1,T2,T3,T4,T5,T6,T7,T8,DI tool
    class FC,CV,MD,EN,D1 data
    class R resp
    class Q,Q2 entry
