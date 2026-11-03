"""
dashboard/app.py
================
Full Interactive Plotly Dash Application for:
“AI-Driven Adaptive Multi-Objective CPU-GPU Workload Scheduling and Resource Optimization for Cloud HPC”

Launch command:
    python dashboard/app.py
Access via web browser at:
    http://localhost:8050
"""

import sys
import time
from pathlib import Path
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.api import DashboardAPI
from profiler.workload_configs import WORKLOAD_CONFIGS
from scheduler.strategy_comparison import StrategyComparison

api = DashboardAPI()
app = dash.Dash(__name__, title="AI CPU-GPU Cloud HPC Scheduler", suppress_callback_exceptions=True)

# Application Layout with Tabs
app.layout = html.Div(style={"fontFamily": "Segoe UI, Arial, sans-serif", "backgroundColor": "#0f172a", "color": "#f8fafc", "minHeight": "100vh", "padding": "20px"}, children=[
    
    # Header Banner
    html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "borderBottom": "2px solid #334155", "paddingBottom": "15px", "marginBottom": "20px"}, children=[
        html.Div([
            html.H1("AI-Driven Adaptive CPU-GPU Workload Scheduler", style={"fontSize": "26px", "fontWeight": "700", "margin": "0", "color": "#38bdf8"}),
            html.P("Multi-Objective Resource Optimization & Live Telemetry Dashboard for Cloud HPC", style={"margin": "5px 0 0 0", "color": "#94a3b8", "fontSize": "14px"}),
        ]),
        html.Div(style={"textAlign": "right"}, children=[
            html.Span("Host: Windows x86_64", style={"backgroundColor": "#1e293b", "padding": "6px 12px", "borderRadius": "6px", "fontSize": "12px", "marginRight": "10px", "border": "1px solid #475569"}),
            html.Span("Target: NVIDIA RTX 5050 / Tesla T4", style={"backgroundColor": "#065f46", "padding": "6px 12px", "borderRadius": "6px", "fontSize": "12px", "color": "#34d399", "fontWeight": "600"}),
        ])
    ]),

    # Main Tab Navigation
    dcc.Tabs(id="main-tabs", value="tab-telemetry", style={"marginBottom": "20px"}, colors={"border": "#334155", "primary": "#38bdf8", "background": "#1e293b"}, children=[
        dcc.Tab(label="1. Live Telemetry & Hardware", value="tab-telemetry", style={"padding": "12px", "fontWeight": "600"}),
        dcc.Tab(label="2. Rodinia Benchmarks", value="tab-benchmarks", style={"padding": "12px", "fontWeight": "600"}),
        dcc.Tab(label="3. ML Placement Models", value="tab-ml", style={"padding": "12px", "fontWeight": "600"}),
        dcc.Tab(label="4. Interactive Scheduler", value="tab-scheduler", style={"padding": "12px", "fontWeight": "600"}),
        dcc.Tab(label="5. Strategy Comparison", value="tab-strategies", style={"padding": "12px", "fontWeight": "600"}),
    ]),

    # Content Container
    html.Div(id="tab-content"),

    # Interval timer for live resource polling (2 seconds)
    dcc.Interval(id="telemetry-interval", interval=2000, n_intervals=0),
])


# ==============================================================================
# TAB 1: LIVE HARDWARE TELEMETRY
# ==============================================================================
def render_telemetry_tab():
    return html.Div([
        html.Div(id="live-telemetry-metrics"),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginTop": "20px"}, children=[
            html.Div(style={"backgroundColor": "#1e293b", "padding": "20px", "borderRadius": "8px", "border": "1px solid #334155"}, children=[
                html.H3("Host CPU Core Utilization (%)", style={"margin": "0 0 15px 0", "fontSize": "16px", "color": "#38bdf8"}),
                dcc.Graph(id="cpu-cores-graph", config={"displayModeBar": False}),
            ]),
            html.Div(style={"backgroundColor": "#1e293b", "padding": "20px", "borderRadius": "8px", "border": "1px solid #334155"}, children=[
                html.H3("GPU Hardware Telemetry Probe", style={"margin": "0 0 15px 0", "fontSize": "16px", "color": "#38bdf8"}),
                html.Div(id="gpu-status-card"),
            ]),
        ])
    ])


@app.callback(
    [Output("live-telemetry-metrics", "children"),
     Output("cpu-cores-graph", "figure"),
     Output("gpu-status-card", "children")],
    [Input("telemetry-interval", "n_intervals")],
    [State("main-tabs", "value")]
)
def update_live_telemetry(n, current_tab):
    if current_tab != "tab-telemetry":
        raise dash.exceptions.PreventUpdate

    snap = api.get_live_resources()
    cpu = snap["cpu"]
    gpu = snap["gpu"]

    # 1. Live 19-Workload Benchmark Stream Panel
    bench = api.get_live_benchmark_status()
    is_running = bench.get("is_running", False)
    completed_cnt = bench.get("completed_count", 0)
    total_cnt = bench.get("total_workloads", 19)
    pct = bench.get("progress_percent", 0.0)
    cur_name = bench.get("current_workload_name", "None")
    cur_domain = bench.get("current_workload_domain", "General")
    cur_dev = bench.get("current_device", "GPU")
    cur_desc = bench.get("current_desc", "")
    elapsed = bench.get("overall_elapsed_s", 0.0)
    completed_list = bench.get("completed_workloads", [])

    all_names = ['bfs', 'cfd', 'hotspot', 'lud', 'kmeans', 'srad', 'nn', 'nw', 'backprop', 'pathfinder', 'particlefilter', 'heartwall', 'b_tree', 'leukocyte', 'streamcluster', 'hybridsort', 'lavaMD', 'gaussian', 'mummergpu']
    completed_names = {c["name"] for c in completed_list}

    if is_running:
        badge_style = {"backgroundColor": "#065f46", "color": "#34d399", "padding": "5px 14px", "borderRadius": "6px", "fontWeight": "700", "fontSize": "12px", "border": "1px solid #059669"}
        header_text = f"LIVE RODINIA STREAM: [{completed_cnt + 1}/{total_cnt}] {cur_name.upper()} ({cur_domain})"
        status_sub = f"Active Kernel: {cur_desc} | Dispatched to: {cur_dev} | Elapsed: {elapsed:.1f}s"
    elif completed_cnt >= total_cnt and total_cnt > 0:
        badge_style = {"backgroundColor": "#1e3a8a", "color": "#60a5fa", "padding": "5px 14px", "borderRadius": "6px", "fontWeight": "700", "fontSize": "12px", "border": "1px solid #2563eb"}
        header_text = f"19/19 RODINIA WORKLOADS PROFILED SUCCESSFULLY ({elapsed:.1f}s)"
        status_sub = "Full 19-workload scientific suite profiled across NVIDIA RTX 5050 GPU and Host CPU"
    else:
        badge_style = {"backgroundColor": "#334155", "color": "#94a3b8", "padding": "5px 14px", "borderRadius": "6px", "fontWeight": "600", "fontSize": "12px"}
        header_text = "Rodinia 19-Workload Scientific Suite Ready"
        status_sub = "19 Scientific Workloads configured across GPU (CUDA) and CPU (16 Cores)"

    pills = []
    for w in all_names:
        if w == cur_name and is_running:
            p_style = {"backgroundColor": "#0284c7", "color": "#ffffff", "padding": "4px 9px", "borderRadius": "4px", "fontSize": "11px", "fontWeight": "700", "border": "1px solid #38bdf8"}
            p_text = f"▶ {w.upper()}"
        elif w in completed_names:
            p_style = {"backgroundColor": "#064e3b", "color": "#34d399", "padding": "4px 9px", "borderRadius": "4px", "fontSize": "11px", "fontWeight": "600"}
            p_text = f"✓ {w}"
        else:
            p_style = {"backgroundColor": "#1e293b", "color": "#64748b", "padding": "4px 9px", "borderRadius": "4px", "fontSize": "11px", "border": "1px solid #334155"}
            p_text = w
        pills.append(html.Span(p_text, style=p_style))

    banner_card = html.Div(style={"backgroundColor": "#1e293b", "padding": "16px 20px", "borderRadius": "8px", "border": "1px solid #38bdf8" if is_running else "1px solid #334155", "marginBottom": "20px"}, children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "10px"}, children=[
            html.Div([
                html.Span(header_text, style={"fontSize": "16px", "fontWeight": "700", "color": "#f8fafc", "marginRight": "12px"}),
                html.Span(status_sub, style={"fontSize": "12px", "color": "#94a3b8"}),
            ]),
            html.Span(f"{pct:.1f}% ({completed_cnt}/{total_cnt})", style=badge_style),
        ]),
        html.Div(style={"backgroundColor": "#0f172a", "borderRadius": "6px", "height": "8px", "overflow": "hidden", "marginBottom": "12px"}, children=[
            html.Div(style={"width": f"{pct}%", "height": "100%", "backgroundColor": "#38bdf8" if is_running else "#10b981", "transition": "width 0.4s ease-in-out"})
        ]),
        html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "6px"}, children=pills)
    ])

    # 2. Metric Cards
    metrics_row = html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "15px"}, children=[
        html.Div(style={"backgroundColor": "#1e293b", "padding": "18px", "borderRadius": "8px", "border": "1px solid #334155"}, children=[
            html.Div("Overall CPU Utilization", style={"color": "#94a3b8", "fontSize": "12px"}),
            html.Div(f"{cpu['cpu_percent']}%", style={"fontSize": "28px", "fontWeight": "700", "color": "#38bdf8", "marginTop": "5px"}),
            html.Div(f"{cpu['cpu_count_logical']} Cores Active ({cpu.get('cpu_freq_current_mhz') or 0:.0f} MHz)", style={"color": "#64748b", "fontSize": "11px", "marginTop": "4px"}),
        ]),
        html.Div(style={"backgroundColor": "#1e293b", "padding": "18px", "borderRadius": "8px", "border": "1px solid #334155"}, children=[
            html.Div("RAM Memory Usage", style={"color": "#94a3b8", "fontSize": "12px"}),
            html.Div(f"{cpu['memory_percent']}%", style={"fontSize": "28px", "fontWeight": "700", "color": "#a855f7", "marginTop": "5px"}),
            html.Div(f"{cpu['memory_used_gb']} GB used / {cpu['memory_total_gb']} GB total", style={"color": "#64748b", "fontSize": "11px", "marginTop": "4px"}),
        ]),
        html.Div(style={"backgroundColor": "#1e293b", "padding": "18px", "borderRadius": "8px", "border": "1px solid #334155"}, children=[
            html.Div("Available Memory", style={"color": "#94a3b8", "fontSize": "12px"}),
            html.Div(f"{cpu['memory_available_gb']} GB", style={"fontSize": "28px", "fontWeight": "700", "color": "#10b981", "marginTop": "5px"}),
            html.Div("Host headroom available", style={"color": "#64748b", "fontSize": "11px", "marginTop": "4px"}),
        ]),
        html.Div(style={"backgroundColor": "#1e293b", "padding": "18px", "borderRadius": "8px", "border": "1px solid #334155"}, children=[
            html.Div("GPU Hardware State", style={"color": "#94a3b8", "fontSize": "12px"}),
            html.Div("ONLINE" if gpu["gpu_available"] else "UNAVAILABLE",
                     style={"fontSize": "24px", "fontWeight": "700", "color": "#34d399" if gpu["gpu_available"] else "#f43f5e", "marginTop": "5px"}),
            html.Div("NVIDIA NVML / nvidia-smi probe", style={"color": "#64748b", "fontSize": "11px", "marginTop": "4px"}),
        ]),
    ])

    combined_telemetry = html.Div([banner_card, metrics_row])

    # 3. CPU per-core Bar Chart
    core_labels = [f"Core {i}" for i in range(len(cpu["per_cpu_percent"]))]
    fig_cores = px.bar(
        x=core_labels,
        y=cpu["per_cpu_percent"],
        labels={"x": "Core", "y": "Utilization (%)"},
        color=cpu["per_cpu_percent"],
        color_continuous_scale="Blues"
    )
    fig_cores.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        margin=dict(l=20, r=20, t=10, b=30),
        yaxis=dict(range=[0, 100]),
        coloraxis_showscale=False
    )

    # 4. GPU Card
    if gpu["gpu_available"]:
        util = gpu.get('gpu_utilization_percent', 0.0) or 0.0
        mem_used = gpu.get('gpu_memory_used_mb', 0.0) or 0.0
        mem_total = gpu.get('gpu_memory_total_mb', 8151.0) or 8151.0
        mem_pct = gpu.get('gpu_memory_percent', 0.0) or round((mem_used / max(1.0, mem_total)) * 100.0, 1)
        temp = gpu.get('gpu_temperature_c', 0.0) or 0.0

        is_active = util > 2.0 or mem_used > 50.0
        status_color = "#34d399" if is_active else "#94a3b8"
        status_label = "ACTIVE BENCHMARK RUNNING" if is_active else "IDLE (Ready for Workload)"

        gpu_card = html.Div(children=[
            html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "15px"}, children=[
                html.Span("NVIDIA GeForce RTX 5050 Laptop GPU", style={"fontWeight": "700", "fontSize": "15px", "color": "#f8fafc"}),
                html.Span(status_label, style={"backgroundColor": "#065f46" if is_active else "#334155", "color": status_color, "padding": "4px 10px", "borderRadius": "4px", "fontSize": "11px", "fontWeight": "700"}),
            ]),
            
            # GPU Utilization Bar
            html.Div(style={"marginBottom": "14px"}, children=[
                html.Div(style={"display": "flex", "justifyContent": "space-between", "marginBottom": "5px"}, children=[
                    html.Span("Compute Utilization", style={"fontSize": "13px", "color": "#94a3b8"}),
                    html.Span(f"{util:.1f}%", style={"fontSize": "14px", "fontWeight": "700", "color": "#38bdf8" if util < 70 else "#f59e0b"}),
                ]),
                html.Div(style={"backgroundColor": "#0f172a", "borderRadius": "6px", "height": "12px", "overflow": "hidden"}, children=[
                    html.Div(style={"width": f"{min(100.0, max(0.0, util))}%", "height": "100%", "backgroundColor": "#38bdf8" if util < 70 else "#f59e0b", "transition": "width 0.5s ease-in-out"})
                ])
            ]),

            # GPU VRAM Bar
            html.Div(style={"marginBottom": "14px"}, children=[
                html.Div(style={"display": "flex", "justifyContent": "space-between", "marginBottom": "5px"}, children=[
                    html.Span("Dedicated VRAM Allocation", style={"fontSize": "13px", "color": "#94a3b8"}),
                    html.Span(f"{mem_used:.1f} MB / {mem_total:.0f} MB ({mem_pct:.1f}%)", style={"fontSize": "13px", "fontWeight": "600", "color": "#a855f7"}),
                ]),
                html.Div(style={"backgroundColor": "#0f172a", "borderRadius": "6px", "height": "12px", "overflow": "hidden"}, children=[
                    html.Div(style={"width": f"{min(100.0, max(0.0, mem_pct))}%", "height": "100%", "backgroundColor": "#a855f7", "transition": "width 0.5s ease-in-out"})
                ])
            ]),

            # Thermal Telemetry
            html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "borderTop": "1px solid #334155", "paddingTop": "10px", "marginTop": "10px"}, children=[
                html.Span(f"Core Temperature: {temp:.1f} °C", style={"fontSize": "13px", "color": "#10b981" if temp < 70 else "#f43f5e", "fontWeight": "600"}),
                html.Span("Engine: NVIDIA CUDA 12/13 Native", style={"fontSize": "11px", "color": "#64748b"}),
            ])
        ])
    else:
        gpu_card = html.Div(style={"padding": "15px", "backgroundColor": "#334155", "borderRadius": "6px"}, children=[
            html.Div("GPU_MONITORING_UNAVAILABLE", style={"color": "#f87171", "fontWeight": "700", "fontSize": "15px", "marginBottom": "8px"}),
            html.P("No physical NVIDIA GPU driver was detected on this local host.", style={"color": "#cbd5e1", "fontSize": "13px", "margin": "0 0 10px 0"}),
        ])

    return combined_telemetry, fig_cores, gpu_card


# ==============================================================================
# TAB 2: RODINIA EMPIRICAL BENCHMARKS (ALL 19 WORKLOADS)
# ==============================================================================
def render_benchmarks_tab():
    benchmarks = api.get_benchmark_summary()
    df = pd.DataFrame(benchmarks)

    fig_time = px.bar(
        df,
        x="workload",
        y=["cpu_median_ms", "gpu_median_ms"],
        barmode="group",
        log_y=True,
        labels={"value": "Median Execution Time (ms, log scale)", "workload": "Rodinia Workload"},
        title="19-Workload CPU vs GPU Execution Time Comparison",
        color_discrete_map={"cpu_median_ms": "#60a5fa", "gpu_median_ms": "#34d399"}
    )
    fig_time.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#1e293b")

    fig_speedup = px.bar(
        df.dropna(subset=["speedup"]),
        x="workload",
        y="speedup",
        color="speedup",
        color_continuous_scale="Viridis",
        labels={"speedup": "Speedup Factor (x)", "workload": "Workload"},
        title="Empirical GPU Speedup Factor Relative to Host CPU Baseline"
    )
    fig_speedup.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Break-even (1.0x)")
    fig_speedup.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#1e293b")

    return html.Div([
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[
            dcc.Graph(figure=fig_time),
            dcc.Graph(figure=fig_speedup),
        ]),
        html.H3("Complete 19-Workload Benchmark Dataset", style={"marginTop": "25px", "fontSize": "16px", "color": "#38bdf8"}),
        dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": i.replace("_", " ").title(), "id": i} for i in df.columns],
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#334155", "color": "#f8fafc", "fontWeight": "bold"},
            style_cell={"backgroundColor": "#1e293b", "color": "#cbd5e1", "fontSize": "13px", "padding": "10px"},
        )
    ])


# ==============================================================================
# TAB 3: MACHINE LEARNING DEVICE CLASSIFICATION
# ==============================================================================
def render_ml_tab():
    metrics = api.get_ml_metrics()
    m_a = metrics.get("model_a", {})
    m_b = metrics.get("model_b", {})
    m_c = metrics.get("model_c", {})
    m_base = metrics.get("majority_baseline", {})

    compare_df = pd.DataFrame([
        {"Model": "Majority Baseline", "Accuracy": m_base.get("accuracy", 0), "Precision": m_base.get("precision_macro", 0), "Recall": m_base.get("recall_macro", 0), "F1-Score": m_base.get("f1_macro", 0)},
        {"Model": "Model A (Random Forest Full)", "Accuracy": m_a.get("accuracy", 0), "Precision": m_a.get("precision_macro", 0), "Recall": m_a.get("recall_macro", 0), "F1-Score": m_a.get("f1_macro", 0)},
        {"Model": "Model B (RF Ablated)", "Accuracy": m_b.get("accuracy", 0), "Precision": m_b.get("precision_macro", 0), "Recall": m_b.get("recall_macro", 0), "F1-Score": m_b.get("f1_macro", 0)},
        {"Model": "Model C (Gradient Boosting)", "Accuracy": m_c.get("accuracy", 0), "Precision": m_c.get("precision_macro", 0), "Recall": m_c.get("recall_macro", 0), "F1-Score": m_c.get("f1_macro", 0)},
    ])

    fig_comp = px.bar(
        compare_df,
        x="Model",
        y=["Accuracy", "Precision", "Recall", "F1-Score"],
        barmode="group",
        title="Leave-One-Workload-Out (LOWO) Cross-Validation Performance"
    )
    fig_comp.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#1e293b")

    return html.Div([
        dcc.Graph(figure=fig_comp),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginTop": "20px"}, children=[
            html.Div(style={"backgroundColor": "#1e293b", "padding": "20px", "borderRadius": "8px", "border": "1px solid #334155"}, children=[
                html.H4("Model A Feature Importance (Top Predictors)", style={"color": "#38bdf8", "margin": "0 0 10px 0"}),
                html.Ul(style={"color": "#cbd5e1", "fontSize": "13px", "lineHeight": "1.8"}, children=[
                    html.Li("1. estimated_transfer_size_mib (0.303) — Host-device PCIe bottleneck"),
                    html.Li("2. input_size_mib (0.299) — Problem domain magnitude"),
                    html.Li("3. estimated_compute_intensity (0.126) — FLOPs per memory byte"),
                    html.Li("4. estimated_memory_boundness (0.080) — Cache line reuse ratio"),
                    html.Li("5. workload_type (0.061) — Algorithmic paradigm"),
                ])
            ]),
            html.Div(style={"backgroundColor": "#1e293b", "padding": "20px", "borderRadius": "8px", "border": "1px solid #334155"}, children=[
                html.H4("Zero-Leakage Architectural Guarantee", style={"color": "#38bdf8", "margin": "0 0 10px 0"}),
                html.P("Predictions are executed strictly prior to job execution using intrinsic workload features derived from the Frozen Rubric v1.0. Zero post-execution metrics (runtime, power, speedup) are admitted.", style={"color": "#cbd5e1", "fontSize": "13px"}),
                html.P("Evaluated with Grouped Leave-One-Workload-Out (LOWO) cross-validation across all 7 holdouts to guarantee generalization on unseen scientific workloads.", style={"color": "#94a3b8", "fontSize": "12px"}),
            ])
        ])
    ])


# ==============================================================================
# TAB 4: INTERACTIVE SCHEDULER
# ==============================================================================
def render_scheduler_tab():
    all_wl = list(WORKLOAD_CONFIGS.keys())
    return html.Div([
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 3fr", "gap": "20px"}, children=[
            # Control Panel
            html.Div(style={"backgroundColor": "#1e293b", "padding": "20px", "borderRadius": "8px", "border": "1px solid #334155"}, children=[
                html.H3("Workload Submission", style={"margin": "0 0 15px 0", "fontSize": "16px", "color": "#38bdf8"}),
                html.Label("Select Tasks to Schedule:", style={"fontSize": "12px", "color": "#94a3b8"}),
                dcc.Dropdown(
                    id="sched-workloads-dropdown",
                    options=[{"label": f"{w.upper()}", "value": w} for w in all_wl],
                    value=all_wl,
                    multi=True,
                    style={"color": "#000", "marginBottom": "15px"}
                ),
                html.Label("Optimization Priority Mode:", style={"fontSize": "12px", "color": "#94a3b8"}),
                dcc.RadioItems(
                    id="sched-mode-radio",
                    options=[
                        {"label": " Balanced (Makespan + Energy + Transfer)", "value": "balanced"},
                        {"label": " Speed Priority (Minimize Makespan)", "value": "speed"},
                        {"label": " Green HPC (Minimize Energy)", "value": "energy"},
                        {"label": " Interconnect (Minimize Transfer)", "value": "transfer"},
                    ],
                    value="balanced",
                    style={"fontSize": "13px", "lineHeight": "2.0", "marginBottom": "20px"}
                ),
                html.Button("Run AI Scheduler", id="sched-run-btn", n_clicks=0,
                            style={"backgroundColor": "#0284c7", "color": "#fff", "border": "none", "padding": "12px 20px", "borderRadius": "6px", "cursor": "pointer", "fontWeight": "700", "width": "100%", "fontSize": "14px"}),
            ]),
            # Execution Results
            html.Div(style={"backgroundColor": "#1e293b", "padding": "20px", "borderRadius": "8px", "border": "1px solid #334155"}, children=[
                html.H3("Live Scheduling Decisions & Optimization Plan", style={"margin": "0 0 15px 0", "fontSize": "16px", "color": "#38bdf8"}),
                dcc.Loading(
                    id="sched-loading",
                    type="circle",
                    color="#38bdf8",
                    children=html.Div(id="sched-results-container", children=html.Div(
                        style={"padding": "35px", "textAlign": "center", "color": "#94a3b8"},
                        children=[
                            html.Div("All 19 Rodinia Scientific Workloads are pre-selected.", style={"fontSize": "15px", "fontWeight": "600", "color": "#f8fafc", "marginBottom": "8px"}),
                            html.Div("Click the 'Run AI Scheduler' button on the left to execute the Bayesian multi-objective optimization plan.", style={"fontSize": "13px", "color": "#94a3b8"})
                        ]
                    ))
                )
            ])
        ])
    ])


@app.callback(
    Output("sched-results-container", "children"),
    [Input("sched-run-btn", "n_clicks")],
    [State("sched-workloads-dropdown", "value"),
     State("sched-mode-radio", "value")]
)
def handle_run_scheduler(n_clicks, workloads, mode):
    if not n_clicks or n_clicks == 0:
        return html.Div(
            style={"padding": "35px", "textAlign": "center", "color": "#94a3b8"},
            children=[
                html.Div("All 19 Rodinia Scientific Workloads are pre-selected.", style={"fontSize": "15px", "fontWeight": "600", "color": "#f8fafc", "marginBottom": "8px"}),
                html.Div("Click the 'Run AI Scheduler' button on the left to execute the Bayesian multi-objective optimization plan.", style={"fontSize": "13px", "color": "#94a3b8"})
            ]
        )

    if not workloads:
        return html.Div("Please select at least one workload task.", style={"color": "#f87171", "padding": "20px"})
    
    t_start = time.time()
    res = api.run_scheduling_pipeline(workloads, mode=mode)
    elapsed_solve = time.time() - t_start

    results = res["results"]
    df = pd.DataFrame(results)

    # Format output table
    display_cols = ["workload", "device_assigned", "predicted_device", "mean_s", "energy_joules", "correct_assignment"]
    df_disp = df[display_cols].copy()
    if "mean_s" in df_disp.columns:
        df_disp["mean_s"] = df_disp["mean_s"].round(3)
    if "energy_joules" in df_disp.columns:
        df_disp["energy_joules"] = df_disp["energy_joules"].round(2)
    if "correct_assignment" in df_disp.columns:
        df_disp["correct_assignment"] = df_disp["correct_assignment"].map({True: "YES", False: "NO"})
    df_disp.columns = ["Workload", "Assigned Device", "ML Recommended", "Runtime (s)", "Energy (Joules)", "ML Match"]

    weights = res["bayesian_weights"]
    gpu_cnt = sum(1 for r in results if r.get("device_assigned") == "gpu")
    cpu_cnt = len(results) - gpu_cnt

    summary_banner = html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "15px", "marginBottom": "15px", "padding": "14px", "backgroundColor": "#0f172a", "borderRadius": "6px", "border": "1px solid #334155"}, children=[
        html.Div([html.Span("Tasks: ", style={"color": "#94a3b8", "fontSize": "12px"}), html.Strong(f"{res['tasks_executed']} Workloads", style={"color": "#38bdf8"})]),
        html.Div([html.Span("Device Allocation: ", style={"color": "#94a3b8", "fontSize": "12px"}), html.Strong(f"{gpu_cnt} GPU / {cpu_cnt} CPU", style={"color": "#34d399"})]),
        html.Div([html.Span("Est Makespan: ", style={"color": "#94a3b8", "fontSize": "12px"}), html.Strong(f"{res['workflow_makespan_s']} s", style={"color": "#f59e0b"})]),
        html.Div([html.Span("Total Energy: ", style={"color": "#94a3b8", "fontSize": "12px"}), html.Strong(f"{res['total_energy_j']} Joules", style={"color": "#a855f7"})]),
        html.Div([html.Span("Bayesian Weights (M/E/T): ", style={"color": "#94a3b8", "fontSize": "12px"}), html.Strong(f"{weights[0]:.2f} / {weights[1]:.2f} / {weights[2]:.2f}", style={"color": "#cbd5e1"})]),
        html.Div([html.Span("Solved At: ", style={"color": "#94a3b8", "fontSize": "12px"}), html.Strong(f"{time.strftime('%H:%M:%S')} ({elapsed_solve:.2f}s)", style={"color": "#10b981"})]),
    ])

    # Workload Allocation Chart
    fig_alloc = px.bar(
        df_disp,
        x="Workload",
        y="Runtime (s)",
        color="Assigned Device",
        title="Scheduled Tasks Runtime by Assigned Execution Slot",
        color_discrete_map={"gpu": "#34d399", "cpu": "#38bdf8", "GPU": "#34d399", "CPU": "#38bdf8"}
    )
    fig_alloc.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#1e293b", height=280, margin=dict(l=20, r=20, t=35, b=20))

    table = dash_table.DataTable(
        data=df_disp.to_dict("records"),
        columns=[{"name": i, "id": i} for i in df_disp.columns],
        style_header={"backgroundColor": "#334155", "color": "#f8fafc", "fontWeight": "bold"},
        style_cell={"backgroundColor": "#1e293b", "color": "#cbd5e1", "fontSize": "13px", "padding": "8px"},
        style_data_conditional=[
            {
                "if": {"filter_query": "{Assigned Device} = 'gpu' || {Assigned Device} = 'GPU'"},
                "backgroundColor": "#064e3b",
                "color": "#34d399",
                "fontWeight": "600"
            }
        ]
    )

    return html.Div([summary_banner, dcc.Graph(figure=fig_alloc), table])


# ==============================================================================
# TAB 5: STRATEGY COMPARISON (ALL 19 WORKLOADS)
# ==============================================================================
def render_strategies_tab():
    comp_records = api.get_strategy_comparison()
    df = pd.DataFrame(comp_records)

    fig_makespan = px.bar(
        df,
        x="strategy",
        y="makespan_s",
        color="strategy",
        title="Workflow Makespan Across Scheduling Policies (Lower is Better)",
        labels={"makespan_s": "Makespan (seconds)", "strategy": "Strategy"}
    )
    fig_makespan.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#1e293b", showlegend=False)

    fig_energy = px.bar(
        df,
        x="strategy",
        y="total_energy_j",
        color="strategy",
        title="Total Energy Consumption Across Scheduling Policies (TDP Proxy)",
        labels={"total_energy_j": "Energy (Joules)", "strategy": "Strategy"}
    )
    fig_energy.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#1e293b", showlegend=False)

    df_disp = df.drop(columns=["assignment"], errors="ignore").copy()
    if "makespan_s" in df_disp.columns:
        df_disp["makespan_s"] = df_disp["makespan_s"].round(3)
    if "total_energy_j" in df_disp.columns:
        df_disp["total_energy_j"] = df_disp["total_energy_j"].round(2)
    if "speedup_over_cpu" in df_disp.columns:
        df_disp["speedup_over_cpu"] = df_disp["speedup_over_cpu"].round(2)
    df_disp.columns = [c.replace("_", " ").title() for c in df_disp.columns]

    return html.Div([
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[
            dcc.Graph(figure=fig_makespan),
            dcc.Graph(figure=fig_energy),
        ]),
        html.H3("Multi-Policy Benchmark Summary across 19 Workloads", style={"marginTop": "20px", "fontSize": "16px", "color": "#38bdf8"}),
        dash_table.DataTable(
            data=df_disp.to_dict("records"),
            columns=[{"name": i, "id": i} for i in df_disp.columns],
            style_header={"backgroundColor": "#334155", "color": "#f8fafc", "fontWeight": "bold"},
            style_cell={"backgroundColor": "#1e293b", "color": "#cbd5e1", "fontSize": "13px", "padding": "8px"},
        )
    ])


# Tab Switching Callback
@app.callback(Output("tab-content", "children"), [Input("main-tabs", "value")])
def render_tab_content(tab):
    if tab == "tab-telemetry":
        return render_telemetry_tab()
    elif tab == "tab-benchmarks":
        return render_benchmarks_tab()
    elif tab == "tab-ml":
        return render_ml_tab()
    elif tab == "tab-scheduler":
        return render_scheduler_tab()
    elif tab == "tab-strategies":
        return render_strategies_tab()
    return render_telemetry_tab()


if __name__ == "__main__":
    print("Launching AI Cloud HPC Live Scheduler Dashboard on http://localhost:8050")
    app.run(host="127.0.0.1", port=8050, debug=False)
