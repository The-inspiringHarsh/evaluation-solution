"""Normalisation and validation rules. Validators never change the observed value; they return a
normalised candidate plus pass/fail and human-readable notes that feed the confidence score."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Callable, Optional

# Visually confusable characters (handwriting and OCR): letter -> digit and digit -> letter.
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "|": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"}
TO_ALPHA = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T"}


@dataclass
class Validation:
    normalized: Optional[str]
    passed: Optional[bool]  # None = no applicable rule
    notes: list[str] = field(default_factory=list)
    coerced: bool = False  # positional confusable-character normalisation was applied


def _compact(value: str) -> str:
    return re.sub(r"[\s\-_.,/]", "", value).upper()


def coerce_pattern(value: str, pattern: str) -> tuple[str, list[str]]:
    """Coerce confusable characters by position. ``pattern`` uses A=letter, 9=digit, X=either, 0=literal zero."""
    out, notes = [], []
    for i, (ch, kind) in enumerate(zip(value, pattern)):
        new = ch
        if kind in ("9", "0") and not ch.isdigit() and ch in TO_DIGIT:
            new = TO_DIGIT[ch]
        elif kind == "A" and ch.isdigit() and ch in TO_ALPHA:
            new = TO_ALPHA[ch]
        if new != ch:
            notes.append(f"position {i + 1}: '{ch}'→'{new}'")
        out.append(new)
    out.extend(value[len(pattern) :])
    return "".join(out), notes


# --------------------------------------------------------------------------- Aadhaar (Verhoeff)

_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(number: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def validate_aadhaar(value: str) -> Validation:
    v = _compact(value)
    v, notes = coerce_pattern(v, "9" * 12)
    if not re.fullmatch(r"\d{12}", v):
        return Validation(v, False, notes + ["Aadhaar must be exactly 12 digits"], bool(notes))
    fails = []
    if v[0] in "01":
        fails.append("Aadhaar numbers never start with 0 or 1")
    if not verhoeff_valid(v):
        fails.append("Verhoeff checksum failed")
    formatted = f"{v[:4]} {v[4:8]} {v[8:]}"
    return Validation(formatted, not fails, notes + fails, bool(notes))


# --------------------------------------------------------------------------- PAN / TIN

PAN_HOLDER_TYPES = set("ABCFGHLJPTK")


def validate_pan(value: str) -> Validation:
    v = _compact(value)
    v, notes = coerce_pattern(v, "AAAAA9999A")
    if not re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", v):
        return Validation(v, False, notes + ["PAN must match AAAAA9999A"], bool(notes))
    if v[3] not in PAN_HOLDER_TYPES:
        return Validation(v, False, notes + [f"4th character '{v[3]}' is not a valid PAN holder-type code"], bool(notes))
    return Validation(v, True, notes, bool(notes))


# --------------------------------------------------------------------------- IFSC / account / amount


def validate_ifsc(value: str) -> Validation:
    v = _compact(value)
    v, notes = coerce_pattern(v, "AAAA0XXXXXX")
    if not re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", v):
        return Validation(v, False, notes + ["IFSC must be 4 letters, '0', then 6 alphanumerics (11 chars)"], bool(notes))
    return Validation(v, True, notes, bool(notes))


def validate_account(value: str) -> Validation:
    v = _compact(value)
    v, notes = coerce_pattern(v, "9" * len(v))
    if not v.isdigit():
        return Validation(v, False, notes + ["account number must contain digits only"], bool(notes))
    if not 9 <= len(v) <= 18:
        return Validation(v, False, notes + [f"account number length {len(v)} outside 9-18"], bool(notes))
    return Validation(v, True, notes, bool(notes))  # string keeps leading zeroes


def validate_digits(value: str) -> Validation:
    v = _compact(value)
    v, notes = coerce_pattern(v, "9" * len(v))
    if not v.isdigit():
        return Validation(v, False, notes + ["expected digits only"], bool(notes))
    return Validation(v, True, notes, bool(notes))


def validate_amount(value: str) -> Validation:
    v = value.replace("₹", "").replace("Rs.", "").replace("Rs", "").replace("INR", "").strip()
    v = re.sub(r"[/\-=]+$", "", v).strip()
    compact = v.replace(",", "").replace(" ", "")
    if re.fullmatch(r"\d+(\.\d{1,2})?", compact):
        return Validation(compact, True, [])
    return Validation(compact, False, ["amount is not a valid number"])


_NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1_000, "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "crore": 10_000_000}


def words_to_number(text: str) -> Optional[int]:
    """Parse Indian-English amount words ('Fifty thousand only') into an integer."""
    tokens = re.findall(r"[a-z]+", text.lower())
    total, current, seen = 0, 0, False
    for tok in tokens:
        if tok in {"only", "and", "rupees", "rupee", "rs"}:
            continue
        if tok in _NUM_WORDS:
            current += _NUM_WORDS[tok]
            seen = True
        elif tok == "hundred":
            current = max(current, 1) * 100
            seen = True
        elif tok in _SCALES:
            total += max(current, 1) * _SCALES[tok]
            current, seen = 0, True
        else:
            return None
    return total + current if seen else None


# --------------------------------------------------------------------------- dates

_DATE_PATTERNS = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y", "%d.%m.%y", "%d %b %Y", "%d %B %Y", "%d/%b/%Y"]


def parse_date(value: str, prefer_past: bool = False, today: Optional[date] = None) -> Optional[date]:
    today = today or date.today()
    text = re.sub(r"\s*([/\-.])\s*", r"\1", value.strip())
    text = re.sub(r"\s+", " ", text)
    for fmt in _DATE_PATTERNS:
        try:
            parsed = datetime.strptime(text, fmt).date()
        except ValueError:
            continue
        if "%y" in fmt and prefer_past and parsed > today:
            parsed = parsed.replace(year=parsed.year - 100)
        return parsed
    digits = re.sub(r"\D", "", value)
    if len(digits) == 8:  # DDMMYYYY written in boxes
        try:
            return datetime.strptime(digits, "%d%m%Y").date()
        except ValueError:
            return None
    return None


def _validate_date(value: str, kind: str) -> Validation:
    cleaned = value.upper().replace("O", "0").replace("I", "1").replace("L", "1")
    coerced = cleaned != value.upper()
    notes = ["confusable characters in date coerced to digits"] if coerced else []
    parsed = parse_date(cleaned, prefer_past=kind in {"date_past", "date_form"})
    if parsed is None:
        return Validation(None, False, notes + ["not a valid calendar date (expected DD/MM/YYYY)"], coerced)
    today = date.today()
    if kind == "date_past" and parsed > today:
        notes.append("date is in the future but should be in the past")
        return Validation(parsed.isoformat(), False, notes, coerced)
    if kind == "date_form" and not (date(2000, 1, 1) <= parsed <= today.replace(year=today.year + 1)):
        notes.append("form date outside the plausible range")
        return Validation(parsed.isoformat(), False, notes, coerced)
    if parsed.year < 1900:
        return Validation(parsed.isoformat(), False, notes + ["year before 1900"], coerced)
    return Validation(parsed.isoformat(), True, notes, coerced)


# --------------------------------------------------------------------------- DL / passport / MRZ

_STATE_CODES = set(
    "AN AP AR AS BR CH CG DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ NL OD OR PB PY RJ SK TN TR TS UK UP WB".split()
)


def validate_dl(value: str) -> Validation:
    v = _compact(value)
    v, notes = coerce_pattern(v, "AA99" + "9" * 11)
    m = re.fullmatch(r"([A-Z]{2})(\d{2})(\d{4})(\d{7})", v)
    if not m:
        return Validation(v, False, notes + ["DL number must be SS RR YYYY NNNNNNN (15 chars)"], bool(notes))
    fails = []
    if m.group(1) not in _STATE_CODES:
        fails.append(f"unknown state code {m.group(1)}")
    year = int(m.group(3))
    if not 1950 <= year <= date.today().year:
        fails.append(f"implausible issue year {year}")
    formatted = f"{m.group(1)}{m.group(2)} {m.group(3)} {m.group(4)}"
    return Validation(formatted, not fails, notes + fails, bool(notes))


def validate_passport(value: str) -> Validation:
    v = _compact(value)
    v, notes = coerce_pattern(v, "A9999999")
    if not re.fullmatch(r"[A-Z]\d{7}", v):
        return Validation(v, False, notes + ["Indian passport number must be 1 letter + 7 digits"], bool(notes))
    return Validation(v, True, notes, bool(notes))


def mrz_check_digit(data: str) -> int:
    weights = (7, 3, 1)
    total = 0
    for i, ch in enumerate(data):
        if ch.isdigit():
            val = int(ch)
        elif ch.isalpha():
            val = ord(ch.upper()) - 55
        else:  # '<'
            val = 0
        total += val * weights[i % 3]
    return total % 10


def parse_mrz_line2(line: str) -> dict[str, str]:
    """Split a TD3 MRZ line 2 into its components (no validation)."""
    return {
        "document_number": line[0:9],
        "document_number_check": line[9],
        "nationality": line[10:13],
        "birth_date": line[13:19],
        "birth_date_check": line[19],
        "sex": line[20],
        "expiry_date": line[21:27],
        "expiry_date_check": line[27],
        "personal_number": line[28:42],
        "personal_number_check": line[42],
        "composite_check": line[43],
    }


def validate_mrz2(value: str) -> Validation:
    v = re.sub(r"\s", "", value.upper()).replace("«", "<").replace("‹", "<")
    if len(v) != 44:
        return Validation(v, False, [f"MRZ line 2 must be 44 characters, got {len(v)}"])
    p = parse_mrz_line2(v)
    fails = []
    checks = [
        ("document number", p["document_number"], p["document_number_check"]),
        ("birth date", p["birth_date"], p["birth_date_check"]),
        ("expiry date", p["expiry_date"], p["expiry_date_check"]),
    ]
    for name, data, check in checks:
        if not check.isdigit() or mrz_check_digit(data) != int(check):
            fails.append(f"MRZ {name} check digit mismatch (expected {mrz_check_digit(data)}, found '{check}')")
    pn_check = p["personal_number_check"]
    expected_pn = mrz_check_digit(p["personal_number"])
    if not (pn_check == "<" and set(p["personal_number"]) <= {"<"}) and (not pn_check.isdigit() or int(pn_check) != expected_pn):
        fails.append(f"MRZ personal number check digit mismatch (expected {expected_pn}, found '{pn_check}')")
    composite_data = v[0:10] + v[13:20] + v[21:43]
    if not p["composite_check"].isdigit() or mrz_check_digit(composite_data) != int(p["composite_check"]):
        fails.append(f"MRZ composite check digit mismatch (expected {mrz_check_digit(composite_data)}, found '{p['composite_check']}')")
    return Validation(v, not fails, fails)


def mrz_date_to_iso(yymmdd: str, prefer_past: bool) -> Optional[str]:
    try:
        parsed = datetime.strptime(yymmdd, "%y%m%d").date()
    except ValueError:
        return None
    if prefer_past and parsed > date.today():
        parsed = parsed.replace(year=parsed.year - 100)
    if not prefer_past and parsed.year < 2000 and parsed < date.today().replace(year=date.today().year - 20):
        parsed = parsed.replace(year=parsed.year + 100)
    return parsed.isoformat()


# --------------------------------------------------------------------------- text-ish fields


def validate_name(value: str) -> Validation:
    v = re.sub(r"\s+", " ", value).strip().strip(".,")
    if not re.search(r"[A-Za-z]", v):
        return Validation(v, False, ["name contains no letters"])
    if re.search(r"\d", v):
        return Validation(v, False, ["name contains digits"])
    return Validation(v, True, [])


def validate_place(value: str) -> Validation:
    v = re.sub(r"\s+", " ", value).strip().strip(".,")
    if not re.fullmatch(r"[A-Za-z][A-Za-z .,'\-]*", v):
        return Validation(v, False, ["place name contains unexpected characters"])
    return Validation(v, True, [])


def validate_text(value: str) -> Validation:
    v = re.sub(r"\s+", " ", value).strip().strip(".,")
    return Validation(v, None, [])


def validate_address(value: str) -> Validation:
    v = re.sub(r"\s+", " ", value.replace("\n", ", ")).strip()
    pin = re.search(r"\b\d{6}\b", v)
    if not pin:
        return Validation(v, False, ["address has no 6-digit PIN code"])
    return Validation(v, True, [])


def validate_checkbox(value: str, options: tuple[str, ...]) -> Validation:
    """Map the selected option(s) onto the canonical option list; exactly one must be selected."""
    parts = [p.strip() for p in re.split(r"[;|]", value) if p.strip()]
    matched = []
    for part in parts:
        for opt in options:
            if opt.lower()[:12] in part.lower() or part.lower()[:12] in opt.lower():
                matched.append(opt)
                break
    matched = list(dict.fromkeys(matched))
    if len(matched) == 1:
        return Validation(matched[0], True, [])
    if not matched:
        return Validation(value.strip(), False, ["selected option does not match any printed option"])
    return Validation("; ".join(matched), False, [f"{len(matched)} options appear selected; exactly one expected"])


VALIDATORS: dict[str, Callable[[str], Validation]] = {
    "aadhaar": validate_aadhaar,
    "pan": validate_pan,
    "ifsc": validate_ifsc,
    "account": validate_account,
    "digits": validate_digits,
    "amount": validate_amount,
    "date": lambda v: _validate_date(v, "date"),
    "date_past": lambda v: _validate_date(v, "date_past"),
    "date_form": lambda v: _validate_date(v, "date_form"),
    "dl": validate_dl,
    "passport": validate_passport,
    "mrz2": validate_mrz2,
    "name": validate_name,
    "place": validate_place,
    "text": validate_text,
    "address": validate_address,
}


def validate(validator: Optional[str], value: str, options: tuple[str, ...] = ()) -> Validation:
    """Run the named validator. Unknown/None validators return the trimmed value with no verdict."""
    if validator == "checkbox":
        return validate_checkbox(value, options)
    fn = VALIDATORS.get(validator or "text", validate_text)
    return fn(value)


def comparison_key(validator: Optional[str], value: Optional[str], options: tuple[str, ...] = ()) -> Optional[str]:
    """Key used to decide whether two independent reads agree (format-aware, case/space-insensitive)."""
    if value is None or not str(value).strip():
        return None
    result = validate(validator, str(value), options)
    base = result.normalized if result.normalized is not None else str(value)
    return re.sub(r"[\s.,]", "", base).upper()
