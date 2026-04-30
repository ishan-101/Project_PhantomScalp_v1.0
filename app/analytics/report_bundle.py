# analytics/report_bundle.py
from __future__ import annotations
import json, base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import plotly.graph_objects as go

@dataclass
class FigureSpec:
    name: str
    fig: go.Figure
    description: str = ""

@dataclass
class TableSpec:
    name: str
    df: pd.DataFrame
    description: str = ""
    as_csv: bool = True

@dataclass
class ReportBundle:
    outdir: Path
    title: str = "Project_PhantomScalp v0.2 Backtest Report"
    meta: Dict[str, Any] = field(default_factory=dict)
    figures: List[FigureSpec] = field(default_factory=list)
    tables: List[TableSpec] = field(default_factory=list)
    attachments: Dict[str, bytes] = field(default_factory=dict)  # name -> bytes

    def add_meta(self, **kwargs) -> "ReportBundle":
        self.meta.update(kwargs); return self

    def add_figure(self, name: str, fig: go.Figure, description: str = "") -> "ReportBundle":
        self.figures.append(FigureSpec(name, fig, description)); return self

    def add_table(self, name: str, df: pd.DataFrame, description: str = "", as_csv: bool = True) -> "ReportBundle":
        self.tables.append(TableSpec(name, df, description, as_csv)); return self

    def add_attachment(self, name: str, data: bytes) -> "ReportBundle":
        self.attachments[name] = data; return self

    def save(self) -> Dict[str, str]:
        self.outdir.mkdir(parents=True, exist_ok=True)

        # 1) JSON summary (config, KPIs)
        json_path = self.outdir / "summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, default=str)

        # 2) CSV exports
        csv_paths = {}
        for t in self.tables:
            if t.as_csv:
                p = self.outdir / f"{t.name}.csv"
                t.df.to_csv(p, index=False)
                csv_paths[t.name] = str(p)

        # 3) Static images (optional) + HTML
        # For HTML we inline Plotly figures; also save standalone PNGs for automation
        img_paths = {}
        try:
            import kaleido  # noqa: F401  # needed for static image export
            for fs in self.figures:
                p = self.outdir / f"{fs.name}.png"
                fs.fig.write_image(str(p), width=1280, height=720, scale=2)
                img_paths[fs.name] = str(p)
        except Exception:
            # Image export is optional; HTML will still work
            pass

        # 4) Save attachments (e.g., confusion matrices as PNG generated elsewhere)
        for name, data in self.attachments.items():
            with open(self.outdir / name, "wb") as f:
                f.write(data)

        # 5) HTML report
        html_path = self.outdir / "report.html"
        html = self._render_html()
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        return {
            "summary_json": str(json_path),
            "report_html": str(html_path),
            **{f"csv::{k}": v for k, v in csv_paths.items()},
            **{f"img::{k}": v for k, v in img_paths.items()},
        }

    def _render_html(self) -> str:
        # Inline all figures as divs with Plotly JSON
        fig_divs = []
        for fs in self.figures:
            fig_json = fs.fig.to_json()
            fig_divs.append(f"""
            <section style="margin:18px 0;">
              <h3>{fs.name}</h3>
              <p style="color:#777;margin-top:-8px;">{fs.description}</p>
              <div id="{fs.name}"></div>
              <script>
                (function() {{
                  var spec = {fig_json};
                  Plotly.newPlot('{fs.name}', spec.data, spec.layout, {{responsive:true, displaylogo:false}});
                }})();
              </script>
            </section>
            """)

        # Simple tables preview (first 10 rows); full CSV is linked
        table_html = []
        for t in self.tables:
            head = t.df.head(10)
            table_html.append(f"""
            <section style="margin:18px 0;">
              <h3>{t.name}</h3>
              <p style="color:#777;margin-top:-8px;">{t.description}</p>
              <div style="overflow:auto;max-height:320px;border:1px solid #ddd;border-radius:8px;padding:6px;">
                {head.to_html(index=False)}
              </div>
              {'<p><a href="'+t.name+'.csv" download>Download CSV</a></p>' if t.as_csv else ''}
            </section>
            """)

        meta_pretty = json.dumps(self.meta, indent=2, default=str)

        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{self.title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 20px; }}
    h1 {{ font-size: 24px; }}
    h2 {{ margin-top: 28px; }}
    code, pre {{ background:#111; color:#eee; padding:12px; border-radius:8px; display:block; }}
    .kpi {{ display:flex; gap:14px; flex-wrap:wrap; }}
    .kpi div {{ background:#f7f7f7; padding:12px 16px; border-radius:12px; }}
    a {{ text-decoration:none; }}
  </style>
</head>
<body>
  <h1>{self.title}</h1>
  <section>
    <h2>Summary</h2>
    <pre><code>{meta_pretty}</code></pre>
  </section>
  <section>
    <h2>Charts</h2>
    {"".join(fig_divs)}
  </section>
  <section>
    <h2>Tables</h2>
    {"".join(table_html)}
  </section>
  <footer style="margin-top:40px;color:#888;">Generated by Project_PhantomScalp v0.2</footer>
</body>
</html>
"""
