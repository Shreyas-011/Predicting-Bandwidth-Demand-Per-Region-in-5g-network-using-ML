"""
╔══════════════════════════════════════════════════════════════════════╗
║  5G Network Bandwidth Demand Prediction — ML Dashboard (Tkinter)    ║
║  Algorithms : Random Forest · Gradient Boosting · SVR · KNN        ║
╚══════════════════════════════════════════════════════════════════════╝

Dependencies:
    pip install pandas numpy scikit-learn matplotlib seaborn
"""

# ── std-lib ────────────────────────────────────────────────────────────
import sys, os, threading, warnings
warnings.filterwarnings("ignore")

# ── third-party ────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, mean_absolute_percentage_error)
from sklearn.pipeline import Pipeline

# ══════════════════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════════════════
DARK_BG      = "#0D1117"
PANEL_BG     = "#161B22"
ACCENT_CYAN  = "#00D4FF"
ACCENT_GREEN = "#00FF88"
ACCENT_GOLD  = "#FFD700"
ACCENT_RED   = "#FF4757"
ACCENT_PURPLE= "#BD93F9"
TEXT_WHITE   = "#E6EDF3"
TEXT_GRAY    = "#8B949E"
BORDER       = "#30363D"
CARD_BG      = "#1C2128"

ALGO_COLORS  = {
    "Random Forest":      ACCENT_CYAN,
    "Gradient Boosting":  ACCENT_GREEN,
    "SVR":                ACCENT_GOLD,
    "KNN":                ACCENT_PURPLE,
}

MPL_STYLE = {
    "axes.facecolor":    PANEL_BG,
    "figure.facecolor":  DARK_BG,
    "text.color":        TEXT_WHITE,
    "axes.labelcolor":   TEXT_WHITE,
    "xtick.color":       TEXT_GRAY,
    "ytick.color":       TEXT_GRAY,
    "axes.edgecolor":    BORDER,
    "grid.color":        BORDER,
    "grid.alpha":        0.5,
    "legend.facecolor":  CARD_BG,
    "legend.edgecolor":  BORDER,
}
plt.rcParams.update(MPL_STYLE)

# ══════════════════════════════════════════════════════════════════════
#  DATA & ML PIPELINE
# ══════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_special_event",
    "active_users_k", "latency_ms", "signal_strength_dbm", "packet_loss_pct",
    "iot_devices", "video_ratio", "slicing_factor", "tower_load_pct",
    "temperature_c",
    # one-hot encoded
    "reg_Mumbai_Urban", "reg_Delhi_Metro", "reg_Bangalore_Tech",
    "reg_Chennai_Port", "reg_Kolkata_Central", "reg_Hyderabad_IT",
    "reg_Pune_Industrial", "reg_Ahmedabad_Commercial",
    "reg_Jaipur_Tourist", "reg_Surat_Trade",
    "net_5G_SA", "net_5G_NSA", "net_5G_mmWave",
]
TARGET_COL = "bandwidth_gbps"

ALGORITHMS = {
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestRegressor(n_estimators=200, max_depth=15,
                                         random_state=42, n_jobs=-1))
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                              max_depth=6, random_state=42))
    ]),
    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  SVR(kernel="rbf", C=100, epsilon=0.5, gamma="scale"))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  KNeighborsRegressor(n_neighbors=10, weights="distance",
                                        algorithm="ball_tree", n_jobs=-1))
    ]),
}

# ══════════════════════════════════════════════════════════════════════
#  HELPER
# ══════════════════════════════════════════════════════════════════════
def metrics_dict(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE%": mape}


# ══════════════════════════════════════════════════════════════════════
#  APPLICATION
# ══════════════════════════════════════════════════════════════════════
class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("5G Bandwidth Demand Predictor — ML Dashboard")
        self.geometry("1380x820")
        self.minsize(1100, 700)
        self.configure(bg=DARK_BG)

        # state
        self.df          = None
        self.X_train     = self.X_test = self.y_train = self.y_test = None
        self.results     = {}   # algo → {model, metrics, y_pred}
        self.active_plot = tk.StringVar(value="Actual vs Predicted")
        self.status_var  = tk.StringVar(value="⬡  Load a dataset to begin")

        self._build_ui()

    # ── UI skeleton ────────────────────────────────────────────────────
    def _build_ui(self):
        self._header()
        body = tk.Frame(self, bg=DARK_BG)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        body.columnconfigure(0, weight=0, minsize=300)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._left_panel(body)
        self._right_panel(body)
        self._status_bar()

    def _header(self):
        hdr = tk.Frame(self, bg=PANEL_BG, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="⬡", font=("Consolas", 22), fg=ACCENT_CYAN,
                 bg=PANEL_BG).pack(side="left", padx=(18, 6))
        tk.Label(hdr, text="5G Bandwidth Demand Predictor",
                 font=("Consolas", 15, "bold"), fg=TEXT_WHITE,
                 bg=PANEL_BG).pack(side="left")
        tk.Label(hdr, text="ML-Powered · Multi-Region · 4 Algorithms",
                 font=("Consolas", 9), fg=TEXT_GRAY, bg=PANEL_BG).pack(
                 side="left", padx=14, pady=(6, 0))

        # top-right controls
        ctrl = tk.Frame(hdr, bg=PANEL_BG)
        ctrl.pack(side="right", padx=18)
        self._btn(ctrl, "📂 Load Dataset", self._load_dataset, ACCENT_CYAN
                  ).pack(side="left", padx=4)
        self._btn(ctrl, "▶ Train All Models", self._train_all, ACCENT_GREEN
                  ).pack(side="left", padx=4)
        self._btn(ctrl, "🔄 Reset", self._reset, ACCENT_RED
                  ).pack(side="left", padx=4)

    def _left_panel(self, parent):
        frame = tk.Frame(parent, bg=PANEL_BG, width=300)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        frame.pack_propagate(False)

        self._section_label(frame, "⬡  ALGORITHMS")
        self.algo_vars = {}
        for name, color in ALGO_COLORS.items():
            var = tk.BooleanVar(value=True)
            self.algo_vars[name] = var
            row = tk.Frame(frame, bg=CARD_BG)
            row.pack(fill="x", padx=10, pady=3)
            tk.Checkbutton(row, variable=var, text=name,
                           font=("Consolas", 9, "bold"),
                           fg=color, bg=CARD_BG, selectcolor=DARK_BG,
                           activebackground=CARD_BG, activeforeground=color,
                           ).pack(side="left", padx=8, pady=6)

        self._section_label(frame, "⬡  SETTINGS")
        self._param_slider(frame, "Test Split %", "split_pct", 10, 40, 20)
        self._param_slider(frame, "CV Folds",     "cv_folds",  3,  10,  5)

        self._section_label(frame, "⬡  METRICS")
        self.metrics_frame = tk.Frame(frame, bg=PANEL_BG)
        self.metrics_frame.pack(fill="x", padx=10, pady=4)
        tk.Label(self.metrics_frame, text="Run training to see metrics.",
                 font=("Consolas", 8), fg=TEXT_GRAY, bg=PANEL_BG
                 ).pack(anchor="w", pady=6)

        self._section_label(frame, "⬡  PLOT TYPE")
        plot_types = [
            "Actual vs Predicted",
            "Residuals",
            "Feature Importance",
            "Region Heatmap",
            "Learning Curves",
            "Error Distribution",
        ]
        for pt in plot_types:
            rb = tk.Radiobutton(frame, text=pt, variable=self.active_plot,
                                value=pt, command=self._refresh_plot,
                                font=("Consolas", 8), fg=TEXT_WHITE,
                                bg=PANEL_BG, selectcolor=DARK_BG,
                                activebackground=PANEL_BG,
                                activeforeground=ACCENT_CYAN)
            rb.pack(anchor="w", padx=14, pady=1)

        self._btn(frame, "💾 Export Results CSV", self._export_results,
                  ACCENT_GOLD).pack(pady=10, padx=10, fill="x")

    def _right_panel(self, parent):
        frame = tk.Frame(parent, bg=DARK_BG)
        frame.grid(row=0, column=1, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        # Canvas for plots
        self.fig = plt.Figure(figsize=(10, 6.5), dpi=95)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = tk.Frame(frame, bg=PANEL_BG)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

        self._draw_welcome()

    def _status_bar(self):
        bar = tk.Frame(self, bg=PANEL_BG, height=26)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        tk.Label(bar, textvariable=self.status_var,
                 font=("Consolas", 8), fg=ACCENT_CYAN, bg=PANEL_BG
                 ).pack(side="left", padx=12)

    # ── Widgets ────────────────────────────────────────────────────────
    def _btn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd,
                         font=("Consolas", 8, "bold"),
                         fg=DARK_BG, bg=color, relief="flat",
                         padx=10, pady=4, cursor="hand2",
                         activebackground=TEXT_WHITE, activeforeground=DARK_BG)

    def _section_label(self, parent, text):
        f = tk.Frame(parent, bg=PANEL_BG)
        f.pack(fill="x", padx=10, pady=(12, 2))
        tk.Label(f, text=text, font=("Consolas", 8, "bold"),
                 fg=ACCENT_CYAN, bg=PANEL_BG).pack(side="left")
        tk.Frame(f, bg=BORDER, height=1).pack(side="left", fill="x",
                                               expand=True, padx=(6, 0))

    def _param_slider(self, parent, label, attr, lo, hi, default):
        f = tk.Frame(parent, bg=PANEL_BG)
        f.pack(fill="x", padx=10, pady=2)
        tk.Label(f, text=label, font=("Consolas", 8), fg=TEXT_GRAY,
                 bg=PANEL_BG, width=14, anchor="w").pack(side="left")
        var = tk.IntVar(value=default)
        setattr(self, attr, var)
        lbl = tk.Label(f, textvariable=var, font=("Consolas", 8, "bold"),
                       fg=ACCENT_CYAN, bg=PANEL_BG, width=3)
        lbl.pack(side="right")
        tk.Scale(f, from_=lo, to=hi, variable=var, orient="horizontal",
                 bg=PANEL_BG, fg=ACCENT_CYAN, troughcolor=CARD_BG,
                 highlightthickness=0, showvalue=False,
                 length=120, sliderlength=12).pack(side="left", padx=4)

    # ── Welcome screen ─────────────────────────────────────────────────
    def _draw_welcome(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(DARK_BG)
        ax.axis("off")
        ax.text(0.5, 0.60, "⬡", fontsize=72, color=ACCENT_CYAN,
                ha="center", va="center", transform=ax.transAxes, alpha=0.3)
        ax.text(0.5, 0.50, "5G Bandwidth Demand Predictor",
                fontsize=18, color=TEXT_WHITE, ha="center", va="center",
                transform=ax.transAxes, fontfamily="monospace")
        ax.text(0.5, 0.42,
                "Load the CSV dataset → Train All Models → Explore Plots",
                fontsize=10, color=TEXT_GRAY, ha="center", va="center",
                transform=ax.transAxes, fontfamily="monospace")
        self.canvas.draw()

    # ── Load data ──────────────────────────────────────────────────────
    def _load_dataset(self):
        path = filedialog.askopenfilename(
            title="Select 5G Bandwidth Dataset CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.df = pd.read_csv(path)
            # fill missing one-hot cols with 0
            needed = [c for c in FEATURE_COLS if c not in self.df.columns]
            for c in needed:
                self.df[c] = 0
            self.status_var.set(
                f"✅  Loaded: {os.path.basename(path)}  "
                f"({len(self.df):,} rows · {len(self.df.columns)} cols)")
            self._prepare_data()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _prepare_data(self):
        available = [c for c in FEATURE_COLS if c in self.df.columns]
        X = self.df[available].fillna(0).values
        y = self.df[TARGET_COL].values
        split = self.split_pct.get() / 100
        (self.X_train, self.X_test,
         self.y_train, self.y_test) = train_test_split(
            X, y, test_size=split, random_state=42)
        self.feature_names = available

    # ── Training ───────────────────────────────────────────────────────
    def _train_all(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Load a dataset first.")
            return
        self._prepare_data()
        self.results.clear()
        selected = [n for n, v in self.algo_vars.items() if v.get()]
        if not selected:
            messagebox.showwarning("No Algorithm", "Select at least one algorithm.")
            return
        self.status_var.set("⟳  Training models … please wait")
        self.update()
        threading.Thread(target=self._train_worker,
                         args=(selected,), daemon=True).start()

    def _train_worker(self, selected):
        try:
            for name in selected:
                self.status_var.set(f"⟳  Training {name} …")
                self.update()
                pipe = ALGORITHMS[name]
                pipe.fit(self.X_train, self.y_train)
                y_pred = pipe.predict(self.X_test)
                cv = cross_val_score(pipe, self.X_train, self.y_train,
                                     cv=self.cv_folds.get(),
                                     scoring="r2", n_jobs=-1)
                self.results[name] = {
                    "model":   pipe,
                    "y_pred":  y_pred,
                    "metrics": metrics_dict(self.y_test, y_pred),
                    "cv_r2":   cv,
                }
            self.after(0, self._training_done)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Training Error", str(e)))

    def _training_done(self):
        self.status_var.set(
            f"✅  Training complete — {len(self.results)} model(s) ready")
        self._update_metrics_panel()
        self._refresh_plot()

    # ── Metrics panel ──────────────────────────────────────────────────
    def _update_metrics_panel(self):
        for w in self.metrics_frame.winfo_children():
            w.destroy()

        headers = ["Model", "R²", "MAE", "RMSE", "MAPE%"]
        widths   = [14,      5,    6,     6,      6]
        header_row = tk.Frame(self.metrics_frame, bg=CARD_BG)
        header_row.pack(fill="x", pady=(0, 2))
        for h, w in zip(headers, widths):
            tk.Label(header_row, text=h, font=("Consolas", 7, "bold"),
                     fg=ACCENT_CYAN, bg=CARD_BG, width=w,
                     anchor="w" if h=="Model" else "e"
                     ).pack(side="left", padx=2)

        for name, res in self.results.items():
            m   = res["metrics"]
            row = tk.Frame(self.metrics_frame, bg=PANEL_BG)
            row.pack(fill="x", pady=1)
            color = ALGO_COLORS[name]
            vals  = [name[:13],
                     f"{m['R²']:.3f}",
                     f"{m['MAE']:.2f}",
                     f"{m['RMSE']:.2f}",
                     f"{m['MAPE%']:.1f}"]
            anchors = ["w", "e", "e", "e", "e"]
            for v, w, a in zip(vals, widths, anchors):
                tk.Label(row, text=v, font=("Consolas", 7),
                         fg=color, bg=PANEL_BG, width=w, anchor=a
                         ).pack(side="left", padx=2)

    # ── Plot dispatcher ────────────────────────────────────────────────
    def _refresh_plot(self):
        if not self.results:
            return
        pt = self.active_plot.get()
        dispatch = {
            "Actual vs Predicted":  self._plot_actual_vs_pred,
            "Residuals":            self._plot_residuals,
            "Feature Importance":   self._plot_feature_importance,
            "Region Heatmap":       self._plot_region_heatmap,
            "Learning Curves":      self._plot_learning_curves,
            "Error Distribution":   self._plot_error_dist,
        }
        self.fig.clear()
        dispatch[pt]()
        self.canvas.draw()

    # ─ Plot 1: Actual vs Predicted ─────────────────────────────────────
    def _plot_actual_vs_pred(self):
        n  = len(self.results)
        gs = gridspec.GridSpec(1, n, figure=self.fig, hspace=0.35)
        for i, (name, res) in enumerate(self.results.items()):
            ax    = self.fig.add_subplot(gs[i])
            color = ALGO_COLORS[name]
            y_pred = res["y_pred"]
            ax.scatter(self.y_test, y_pred, alpha=0.35, s=6,
                       color=color, edgecolors="none")
            lo = min(self.y_test.min(), y_pred.min())
            hi = max(self.y_test.max(), y_pred.max())
            ax.plot([lo, hi], [lo, hi], "--", color=TEXT_WHITE, lw=1.2, alpha=0.6)
            ax.set_xlabel("Actual (Gbps)", fontsize=7)
            ax.set_ylabel("Predicted (Gbps)", fontsize=7)
            ax.set_title(f"{name}\nR²={res['metrics']['R²']:.3f}",
                         fontsize=8, color=color, pad=6)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
        self.fig.suptitle("Actual vs Predicted Bandwidth (Gbps)",
                           color=TEXT_WHITE, fontsize=10, y=1.01)

    # ─ Plot 2: Residuals ───────────────────────────────────────────────
    def _plot_residuals(self):
        n  = len(self.results)
        gs = gridspec.GridSpec(2, n, figure=self.fig, hspace=0.55, wspace=0.35)
        for i, (name, res) in enumerate(self.results.items()):
            color  = ALGO_COLORS[name]
            resid  = self.y_test - res["y_pred"]
            ax_top = self.fig.add_subplot(gs[0, i])
            ax_bot = self.fig.add_subplot(gs[1, i])

            ax_top.scatter(res["y_pred"], resid, alpha=0.35, s=5, color=color)
            ax_top.axhline(0, color=TEXT_WHITE, lw=1, linestyle="--", alpha=0.5)
            ax_top.set_xlabel("Predicted", fontsize=6)
            ax_top.set_ylabel("Residual", fontsize=6)
            ax_top.set_title(name, fontsize=7, color=color)
            ax_top.tick_params(labelsize=6)

            ax_bot.hist(resid, bins=30, color=color, alpha=0.7, edgecolor="none")
            ax_bot.set_xlabel("Residual", fontsize=6)
            ax_bot.set_ylabel("Count", fontsize=6)
            ax_bot.tick_params(labelsize=6)
        self.fig.suptitle("Residual Analysis", color=TEXT_WHITE, fontsize=10)

    # ─ Plot 3: Feature Importance ──────────────────────────────────────
    def _plot_feature_importance(self):
        tree_models = {k: v for k, v in self.results.items()
                       if k in ("Random Forest", "Gradient Boosting")}
        if not tree_models:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "Feature Importance only available\nfor tree-based models\n"
                    "(Random Forest / Gradient Boosting)",
                    ha="center", va="center", color=TEXT_GRAY,
                    fontsize=11, transform=ax.transAxes)
            return

        n  = len(tree_models)
        gs = gridspec.GridSpec(1, n, figure=self.fig)
        for i, (name, res) in enumerate(tree_models.items()):
            ax    = self.fig.add_subplot(gs[i])
            color = ALGO_COLORS[name]
            imp   = res["model"].named_steps["model"].feature_importances_
            idx   = np.argsort(imp)[-15:]
            names = [self.feature_names[j] for j in idx]
            ax.barh(names, imp[idx], color=color, alpha=0.8, edgecolor="none")
            ax.set_xlabel("Importance", fontsize=7)
            ax.set_title(f"Feature Importance\n{name}", fontsize=8, color=color)
            ax.tick_params(labelsize=6)
        self.fig.suptitle("Top-15 Feature Importances",
                           color=TEXT_WHITE, fontsize=10)

    # ─ Plot 4: Region Heatmap ─────────────────────────────────────────
    def _plot_region_heatmap(self):
        if "region" not in self.df.columns:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "'region' column not in dataset",
                    ha="center", va="center", color=TEXT_GRAY, fontsize=11,
                    transform=ax.transAxes)
            return

        # Use the best available model
        best_name = max(self.results, key=lambda n: self.results[n]["metrics"]["R²"])
        best_pipe = self.results[best_name]["model"]

        avail = [c for c in FEATURE_COLS if c in self.df.columns]
        preds = best_pipe.predict(self.df[avail].fillna(0).values)

        tmp = self.df.copy()
        tmp["predicted_bw"] = preds
        pivot = tmp.pivot_table(values="predicted_bw",
                                index="region",
                                columns="hour_of_day",
                                aggfunc="mean")

        ax = self.fig.add_subplot(111)
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd",
                    linewidths=0.3, linecolor=DARK_BG,
                    cbar_kws={"label": "Predicted Bandwidth (Gbps)",
                              "shrink": 0.7})
        ax.set_title(f"Avg Predicted Bandwidth by Region × Hour  [{best_name}]",
                     fontsize=9, color=TEXT_WHITE, pad=8)
        ax.set_xlabel("Hour of Day", fontsize=8)
        ax.set_ylabel("Region", fontsize=8)
        ax.tick_params(labelsize=6)

    # ─ Plot 5: Learning Curves ─────────────────────────────────────────
    def _plot_learning_curves(self):
        from sklearn.model_selection import learning_curve
        n  = len(self.results)
        gs = gridspec.GridSpec(1, n, figure=self.fig, wspace=0.3)
        sizes = np.linspace(0.1, 1.0, 6)
        for i, (name, res) in enumerate(self.results.items()):
            ax    = self.fig.add_subplot(gs[i])
            color = ALGO_COLORS[name]
            try:
                train_s, train_sc, val_sc = learning_curve(
                    res["model"], self.X_train, self.y_train,
                    train_sizes=sizes, cv=3, scoring="r2",
                    n_jobs=-1, error_score="raise")
            except Exception:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        color=TEXT_GRAY, transform=ax.transAxes)
                continue
            t_mean = train_sc.mean(axis=1)
            v_mean = val_sc.mean(axis=1)
            ax.plot(train_s, t_mean, "o-", color=color, label="Train", lw=1.5)
            ax.plot(train_s, v_mean, "s--", color=ACCENT_GOLD, label="CV Val", lw=1.5)
            ax.fill_between(train_s, t_mean - train_sc.std(axis=1),
                            t_mean + train_sc.std(axis=1), alpha=0.1, color=color)
            ax.set_xlabel("Training samples", fontsize=6)
            ax.set_ylabel("R² Score", fontsize=6)
            ax.set_title(name, fontsize=7, color=color)
            ax.legend(fontsize=6)
            ax.tick_params(labelsize=6)
            ax.set_ylim(-0.1, 1.05)
        self.fig.suptitle("Learning Curves", color=TEXT_WHITE, fontsize=10)

    # ─ Plot 6: Error Distribution ──────────────────────────────────────
    def _plot_error_dist(self):
        ax = self.fig.add_subplot(111)
        for name, res in self.results.items():
            abs_err = np.abs(self.y_test - res["y_pred"])
            ax.hist(abs_err, bins=40, alpha=0.55,
                    color=ALGO_COLORS[name], label=name, edgecolor="none")
        ax.set_xlabel("Absolute Error (Gbps)", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.set_title("Absolute Error Distribution Across Algorithms",
                     fontsize=10, color=TEXT_WHITE)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # ── Export ─────────────────────────────────────────────────────────
    def _export_results(self):
        if not self.results:
            messagebox.showwarning("Nothing to export", "Train models first.")
            return
        rows = []
        for name, res in self.results.items():
            row = {"Algorithm": name}
            row.update(res["metrics"])
            row["CV_R2_mean"] = res["cv_r2"].mean()
            row["CV_R2_std"]  = res["cv_r2"].std()
            rows.append(row)
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
        if path:
            pd.DataFrame(rows).to_csv(path, index=False)
            messagebox.showinfo("Saved", f"Results exported to:\n{path}")

    # ── Reset ──────────────────────────────────────────────────────────
    def _reset(self):
        self.df      = None
        self.results = {}
        for w in self.metrics_frame.winfo_children():
            w.destroy()
        tk.Label(self.metrics_frame, text="Run training to see metrics.",
                 font=("Consolas", 8), fg=TEXT_GRAY, bg=PANEL_BG
                 ).pack(anchor="w", pady=6)
        self.status_var.set("⬡  Reset — load a new dataset")
        self._draw_welcome()


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = App()
    app.mainloop()
