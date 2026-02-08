from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import yaml

from ui import service

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Strategy Tester Local UI")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

DEFAULT_INDICATOR_LENGTHS = {
    "rsi": 14,
    "ema": 21,
    "adx": 14,
    "supertrend": 7,
}


class IndicatorInput(BaseModel):
    alias: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    source: str = "close"
    length: int | None = None
    multiplier: float | None = None


class StrategyInput(BaseModel):
    starting_cash: float
    position_size_pct: float
    entry_rule: Any
    exit_rule: Any
    price_column: str = "close"
    force_close_end: bool = True
    allow_short: bool = False


class RunRequest(BaseModel):
    symbol: str
    timeframe: str
    start: str
    end: str
    indicators: list[IndicatorInput] = Field(default_factory=list)
    strategy: StrategyInput | None = None


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "form_state": _default_form_state(),
            "result": None,
            "error": None,
        },
    )


@app.post("/run", response_class=HTMLResponse)
async def run_from_form(request: Request) -> HTMLResponse:
    form = await request.form()
    try:
        cfg = _config_from_form_data(form)
        result = service.run_strategy_from_config(cfg)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "form_state": _form_state_from_cfg(cfg),
                "result": result,
                "error": None,
            },
        )
    except Exception as exc:  # pragma: no cover - UI path
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "form_state": _form_state_from_form_data(form),
                "result": None,
                "error": str(exc),
            },
        )


@app.post("/api/run")
async def run_api(payload: RunRequest) -> dict[str, Any]:
    try:
        cfg = _config_from_api_payload(payload)
        return service.run_strategy_from_config(cfg)
    except Exception as exc:  # pragma: no cover - API path
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _config_from_api_payload(payload: RunRequest) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "symbol": payload.symbol.strip().upper(),
        "timeframe": payload.timeframe.strip(),
        "start": payload.start.strip(),
        "end": payload.end.strip(),
        "indicators": _indicators_to_config(payload.indicators),
    }

    if payload.strategy is not None:
        cfg["strategy"] = {
            "starting_cash": payload.strategy.starting_cash,
            "position_size_pct": payload.strategy.position_size_pct,
            "entry_rule": _parse_rule_field(payload.strategy.entry_rule, "entry_rule"),
            "exit_rule": _parse_rule_field(payload.strategy.exit_rule, "exit_rule"),
            "price_column": payload.strategy.price_column,
            "force_close_end": payload.strategy.force_close_end,
            "allow_short": payload.strategy.allow_short,
        }

    return cfg


def _config_from_form_data(form: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "symbol": str(form.get("symbol", "")).strip().upper(),
        "timeframe": str(form.get("timeframe", "")).strip(),
        "start": str(form.get("start", "")).strip(),
        "end": str(form.get("end", "")).strip(),
        "indicators": _indicators_from_form(form),
    }

    use_strategy = str(form.get("use_strategy", "")).lower() in {"on", "true", "1"}
    if use_strategy:
        rule_mode = str(form.get("rule_mode", "simple")).strip().lower() or "simple"
        entry_from_canonical = str(form.get("entry_rule", "")).strip()
        exit_from_canonical = str(form.get("exit_rule", "")).strip()
        entry_simple = str(form.get("entry_rule_simple", "")).strip()
        exit_simple = str(form.get("exit_rule_simple", "")).strip()
        entry_advanced = str(form.get("entry_rule_advanced", "")).strip()
        exit_advanced = str(form.get("exit_rule_advanced", "")).strip()

        if entry_from_canonical or exit_from_canonical:
            entry_raw = entry_from_canonical
            exit_raw = exit_from_canonical
        elif rule_mode == "advanced":
            entry_raw = entry_advanced
            exit_raw = exit_advanced
        else:
            entry_raw = entry_simple
            exit_raw = exit_simple

        entry_rule, exit_rule = _parse_rule_pair(
            entry_raw,
            exit_raw,
        )
        cfg["strategy"] = {
            "starting_cash": _parse_float(form.get("starting_cash"), "starting_cash"),
            "position_size_pct": _parse_float(form.get("position_size_pct"), "position_size_pct"),
            "entry_rule": entry_rule,
            "exit_rule": exit_rule,
            "price_column": str(form.get("price_column", "close")).strip() or "close",
            "force_close_end": str(form.get("force_close_end", "")).lower() in {"on", "true", "1"},
            "allow_short": str(form.get("allow_short", "")).lower() in {"on", "true", "1"},
        }

    return cfg


def _indicators_to_config(indicators: list[IndicatorInput]) -> dict[str, dict[str, Any]]:
    cfg: dict[str, dict[str, Any]] = {}
    for indicator in indicators:
        alias = indicator.alias.strip()
        kind = indicator.kind.strip().lower()
        if alias in cfg:
            raise ValueError(f"Duplicate indicator alias: '{alias}'")

        spec: dict[str, Any] = {
            "kind": kind,
            "source": indicator.source.strip() if indicator.source else "close",
        }

        length = indicator.length
        if length is None and kind in DEFAULT_INDICATOR_LENGTHS:
            length = DEFAULT_INDICATOR_LENGTHS[kind]

        if length is not None:
            spec["length"] = int(length)

        if indicator.multiplier is not None:
            spec["multiplier"] = float(indicator.multiplier)

        cfg[alias] = spec

    return cfg


def _indicators_from_form(form: Any) -> dict[str, dict[str, Any]]:
    aliases = form.getlist("indicator_alias")
    kinds = form.getlist("indicator_kind")
    sources = form.getlist("indicator_source")
    lengths = form.getlist("indicator_length")
    multipliers = form.getlist("indicator_multiplier")

    row_count = max(len(aliases), len(kinds), len(sources), len(lengths), len(multipliers), 0)
    indicators: dict[str, dict[str, Any]] = {}

    for idx in range(row_count):
        alias = aliases[idx].strip() if idx < len(aliases) else ""
        kind = kinds[idx].strip().lower() if idx < len(kinds) else ""
        source = sources[idx].strip().lower() if idx < len(sources) else ""
        length_raw = lengths[idx].strip() if idx < len(lengths) else ""
        multiplier_raw = multipliers[idx].strip() if idx < len(multipliers) else ""

        if not alias and not kind:
            continue
        if not alias or not kind:
            raise ValueError("Each indicator row needs both alias and kind.")
        if alias in indicators:
            raise ValueError(f"Duplicate indicator alias: '{alias}'")

        spec: dict[str, Any] = {
            "kind": kind,
            "source": source or "close",
        }

        if length_raw:
            spec["length"] = _parse_int(length_raw, f"indicator_length[{idx}]")
        elif kind in DEFAULT_INDICATOR_LENGTHS:
            spec["length"] = DEFAULT_INDICATOR_LENGTHS[kind]

        if multiplier_raw:
            spec["multiplier"] = _parse_float(multiplier_raw, f"indicator_multiplier[{idx}]")

        indicators[alias] = spec

    return indicators


def _parse_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number.") from exc


def _parse_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc


def _parse_rule_field(value: Any, label: str) -> Any:
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError(f"{label} must not be empty.")

        parsed = yaml.safe_load(raw)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, str):
            return parsed.strip()

        raise ValueError(
            f"{label} must be a string expression or a YAML mapping (e.g. with 'all'/'any')."
        )

    raise ValueError(f"{label} must be a string expression or a YAML mapping.")


def _parse_rule_pair(entry_raw: str, exit_raw: str) -> tuple[Any, Any]:
    entry_parsed = _parse_rule_field(entry_raw, "entry_rule")
    exit_raw = exit_raw.strip()

    # Convenience path: user pastes full block with both entry/exit into one box.
    if isinstance(entry_parsed, dict) and "entry" in entry_parsed and "exit" in entry_parsed and not exit_raw:
        return (
            _parse_rule_field(entry_parsed["entry"], "entry_rule"),
            _parse_rule_field(entry_parsed["exit"], "exit_rule"),
        )

    exit_parsed = _parse_rule_field(exit_raw, "exit_rule")
    return entry_parsed, exit_parsed


def _rule_to_text(rule: Any) -> str:
    if isinstance(rule, dict):
        return yaml.safe_dump(rule, sort_keys=False).strip()
    if rule is None:
        return ""
    return str(rule)


def _looks_structured_rule_text(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    return ("\n" in text) or normalized.startswith("all:") or normalized.startswith("any:") or normalized.startswith("entry:")


def _simple_rule_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned or _looks_structured_rule_text(cleaned):
        return ""
    return cleaned


def _default_form_state() -> dict[str, Any]:
    end_date = date.today()
    start_date = end_date - timedelta(days=365)

    return {
        "symbol": "SPY",
        "timeframe": "1d",
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "indicators": [
            {"alias": "ema21", "kind": "ema", "length": 21, "source": "close", "multiplier": ""},
            {"alias": "ema50", "kind": "ema", "length": 50, "source": "close", "multiplier": ""},
            {"alias": "rsi14", "kind": "rsi", "length": 14, "source": "close", "multiplier": ""},
        ],
        "rule_mode": "simple",
        "use_strategy": True,
        "strategy": {
            "starting_cash": "10000",
            "position_size_pct": "0.25",
            "entry_rule": "ema21 > ema50",
            "exit_rule": "ema21 < ema50",
            "entry_rule_simple": "ema21 > ema50",
            "exit_rule_simple": "ema21 < ema50",
            "entry_rule_advanced": "ema21 > ema50",
            "exit_rule_advanced": "ema21 < ema50",
            "price_column": "close",
            "force_close_end": True,
            "allow_short": False,
        },
    }


def _form_state_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    indicators = []
    for alias, spec in cfg.get("indicators", {}).items():
        indicators.append(
            {
                "alias": alias,
                "kind": spec.get("kind", ""),
                "length": spec.get("length", ""),
                "source": spec.get("source", "close"),
                "multiplier": spec.get("multiplier", ""),
            }
        )

    strategy_cfg = cfg.get("strategy")
    entry_text = _rule_to_text(strategy_cfg.get("entry_rule", "")) if strategy_cfg else ""
    exit_text = _rule_to_text(strategy_cfg.get("exit_rule", "")) if strategy_cfg else ""
    rule_mode = "advanced" if (_looks_structured_rule_text(entry_text) or _looks_structured_rule_text(exit_text)) else "simple"
    return {
        "symbol": cfg.get("symbol", ""),
        "timeframe": cfg.get("timeframe", ""),
        "start": cfg.get("start", ""),
        "end": cfg.get("end", ""),
        "indicators": indicators or [{"alias": "", "kind": "", "length": "", "source": "close", "multiplier": ""}],
        "rule_mode": rule_mode,
        "use_strategy": bool(strategy_cfg),
        "strategy": {
            "starting_cash": strategy_cfg.get("starting_cash", "") if strategy_cfg else "",
            "position_size_pct": strategy_cfg.get("position_size_pct", "") if strategy_cfg else "",
            "entry_rule": entry_text,
            "exit_rule": exit_text,
            "entry_rule_simple": _simple_rule_text(entry_text),
            "exit_rule_simple": _simple_rule_text(exit_text),
            "entry_rule_advanced": entry_text,
            "exit_rule_advanced": exit_text,
            "price_column": strategy_cfg.get("price_column", "close") if strategy_cfg else "close",
            "force_close_end": strategy_cfg.get("force_close_end", True) if strategy_cfg else True,
            "allow_short": strategy_cfg.get("allow_short", False) if strategy_cfg else False,
        },
    }


def _form_state_from_form_data(form: Any) -> dict[str, Any]:
    aliases = form.getlist("indicator_alias")
    kinds = form.getlist("indicator_kind")
    sources = form.getlist("indicator_source")
    lengths = form.getlist("indicator_length")
    multipliers = form.getlist("indicator_multiplier")

    row_count = max(len(aliases), len(kinds), len(sources), len(lengths), len(multipliers), 1)

    indicators = []
    for idx in range(row_count):
        indicators.append(
            {
                "alias": aliases[idx] if idx < len(aliases) else "",
                "kind": kinds[idx] if idx < len(kinds) else "",
                "source": sources[idx] if idx < len(sources) else "close",
                "length": lengths[idx] if idx < len(lengths) else "",
                "multiplier": multipliers[idx] if idx < len(multipliers) else "",
            }
        )

    return {
        "symbol": str(form.get("symbol", "")).strip(),
        "timeframe": str(form.get("timeframe", "")).strip(),
        "start": str(form.get("start", "")).strip(),
        "end": str(form.get("end", "")).strip(),
        "indicators": indicators,
        "rule_mode": str(form.get("rule_mode", "simple")).strip() or "simple",
        "use_strategy": str(form.get("use_strategy", "")).lower() in {"on", "true", "1"},
        "strategy": {
            "starting_cash": str(form.get("starting_cash", "")).strip(),
            "position_size_pct": str(form.get("position_size_pct", "")).strip(),
            "entry_rule": str(form.get("entry_rule", "")).strip(),
            "exit_rule": str(form.get("exit_rule", "")).strip(),
            "entry_rule_simple": str(form.get("entry_rule_simple", "")).strip(),
            "exit_rule_simple": str(form.get("exit_rule_simple", "")).strip(),
            "entry_rule_advanced": str(form.get("entry_rule_advanced", "")).strip(),
            "exit_rule_advanced": str(form.get("exit_rule_advanced", "")).strip(),
            "price_column": str(form.get("price_column", "close")).strip() or "close",
            "force_close_end": str(form.get("force_close_end", "")).lower() in {"on", "true", "1"},
            "allow_short": str(form.get("allow_short", "")).lower() in {"on", "true", "1"},
        },
    }
