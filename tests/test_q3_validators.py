"""Validators: Aadhaar, PAN, IFSC, dates, MRZ, account numbers, amounts, checkboxes."""

import pytest

from src.question3_documents.validators import (
    comparison_key,
    mrz_check_digit,
    validate,
    validate_mrz2,
    verhoeff_valid,
    words_to_number,
)


def _mrz2(number="L898902C3", nat="UTO", dob="740812", sex="F", exp="120415", personal="ZE184226B<<<<<"):
    def cd(s):
        return str(mrz_check_digit(s))

    line = number + cd(number) + nat + dob + cd(dob) + sex + exp + cd(exp) + personal + cd(personal)
    composite = line[0:10] + line[13:20] + line[21:43]
    return line + cd(composite)


def test_icao_reference_mrz_is_valid():
    # ICAO Doc 9303 specimen
    line = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
    assert validate_mrz2(line).passed


def test_mrz_detects_bad_check_digit_and_length():
    good = _mrz2()
    assert validate_mrz2(good).passed
    bad = good[:9] + str((int(good[9]) + 1) % 10) + good[10:]
    res = validate_mrz2(bad)
    assert res.passed is False and any("document number" in n for n in res.notes)
    assert validate_mrz2(good[:-2]).passed is False


@pytest.mark.parametrize("value,ok", [("2341 2341 2346", True), ("1234 5678 9012", False), ("2341 2341 2345", False), ("234123412", False)])
def test_aadhaar(value, ok):
    assert validate("aadhaar", value).passed is ok


def test_verhoeff():
    assert verhoeff_valid("234123412346")
    assert not verhoeff_valid("234123412345")


def test_aadhaar_visually_clear_but_invalid_is_preserved():
    res = validate("aadhaar", "1234 5678 9012")
    assert res.normalized == "1234 5678 9012" and res.passed is False


@pytest.mark.parametrize(
    "value,ok,norm",
    [
        ("BPQPD3051R", True, "BPQPD3051R"),
        ("bpqpd 3051 r", True, "BPQPD3051R"),
        ("ABCDE1234F", False, "ABCDE1234F"),  # 4th char D is not a valid holder type
        ("ABC1234567", False, None),
    ],
)
def test_pan(value, ok, norm):
    res = validate("pan", value)
    assert res.passed is ok
    if norm:
        assert res.normalized == norm


def test_pan_confusables_are_coerced_and_flagged():
    res = validate("pan", "BPQPD3O5IR")
    assert res.normalized == "BPQPD3051R" and res.coerced and res.notes


@pytest.mark.parametrize("value,ok", [("SBIN0227112", True), ("SBINO227112", True), ("SBIN1227112", False), ("SBIN022711", False)])
def test_ifsc(value, ok):
    assert validate("ifsc", value).passed is ok


def test_ifsc_coercion_marks_review():
    assert validate("ifsc", "SBINO227112").coerced


def test_account_number_keeps_leading_zeros():
    res = validate("account", "0012 3456 7890")
    assert res.passed and res.normalized == "001234567890"
    assert validate("account", "12345").passed is False


def test_amount():
    assert validate("amount", "₹ 50,000").normalized == "50000"
    assert validate("amount", "fifty").passed is False


def test_words_to_number():
    assert words_to_number("Fifty thousand Only") == 50000
    assert words_to_number("One lakh twenty five thousand") == 125000
    assert words_to_number("not a number") is None


@pytest.mark.parametrize(
    "kind,value,ok,iso",
    [
        ("date_past", "18/12/1979", True, "1979-12-18"),
        ("date_form", "26/04/2026", True, "2026-04-26"),
        ("date_form", "26/04/26", True, "2026-04-26"),
        ("date", "31/02/2030", False, None),
        ("date_past", "01/01/2999", False, "2999-01-01"),
        ("date", "14/06/2041", True, "2041-06-14"),
    ],
)
def test_dates(kind, value, ok, iso):
    res = validate(kind, value)
    assert res.passed is ok
    assert res.normalized == iso


def test_dl_and_passport():
    assert validate("dl", "MH12 2021 0001234").passed
    assert validate("dl", "ZZ12 2021 0001234").passed is False
    assert validate("passport", "X1234567").passed
    coerced = validate("passport", "12345678")
    assert coerced.coerced and coerced.normalized == "I2345678"  # confusable flagged for review
    assert validate("passport", "1234567").passed is False


def test_checkbox_requires_exactly_one():
    opts = ("Monthly", "Quarterly", "As & when presented")
    assert validate("checkbox", "As & when presented", opts).normalized == "As & when presented"
    assert validate("checkbox", "Monthly; Quarterly", opts).passed is False
    assert validate("checkbox", "Weekly", opts).passed is False


def test_comparison_key_is_format_aware():
    assert comparison_key("ifsc", "SBIN 0227112") == comparison_key("ifsc", "sbinO227112")
    assert comparison_key("name", "Ashok.") == comparison_key("name", "ASHOK")
    assert comparison_key("text", None) is None
