"""End-to-end pipeline with a scripted vision LLM (no network)."""

import csv
import io
import json

import pytest

from src.question3_documents.extract import CLASSIFY_SCHEMA, _full_page_schema
from src.question3_documents.pipeline import process_files
from src.question3_documents.report import FLAG_COLUMNS, flag_rows, summary_rows, to_csv, write_outputs
from src.question3_documents.schemas import DOC_SPECS, SUPPORTED_TYPES, DocumentResult, DocumentType
from tests.conftest import make_text_image, pdf_bytes, png_bytes

EXPECTED_FIELDS = {
    DocumentType.AADHAAR_CARD: ["aadhaar_number", "full_name", "date_of_birth", "address"],
    DocumentType.PAN_CARD: ["pan_number", "full_name", "fathers_name", "date_of_birth"],
    DocumentType.DRIVING_LICENCE: ["dl_number", "name", "date_of_issue", "valid_till"],
    DocumentType.PASSPORT: ["passport_number", "date_of_birth", "date_of_expiry", "mrz_line_2"],
    DocumentType.NACH_MANDATE: ["bank_account_number", "ifsc_code", "bank_name", "amount_in_figures", "frequency"],
    DocumentType.FATCA_ANNEXURE: ["policy_number", "tin_pan", "fathers_name", "place_of_birth", "nationality"],
    DocumentType.BENEFIT_ILLUSTRATION: ["application_number", "policyholder_name", "date", "place"],
    DocumentType.MORAL_HAZARD: ["application_number", "name_of_life_assured", "nominee_relationship", "date", "place"],
    DocumentType.MULTIPLE_POLICIES: ["proposer_name", "reason_for_multiple_policies", "date", "place"],
    DocumentType.SUITABILITY_PROFILER: ["application_number", "name_of_life_assured", "name_of_agent_sp", "date", "place"],
}


class VisionScript:
    """Answers classification / full-page / crop requests from a scenario description."""

    def __init__(self, doc_type, full_values, crop_values=None, binarized_values=None, groups=None):
        self.doc_type = doc_type
        self.full_values = full_values
        self.crop_values = crop_values or {}
        self.binarized_values = binarized_values or {}
        self.groups = groups

    def __call__(self, system, parts, schema):
        props = schema["properties"]
        if "documents" in props:
            n = len([p for p in parts if hasattr(p, "data")])
            groups = self.groups or [list(range(1, n + 1))]
            return {
                "documents": [{"pages": g, "document_type": self.doc_type.value, "confidence": 0.95, "evidence": "title"} for g in groups]
            }
        if "fields" in props:
            return {
                "fields": [
                    {
                        "field": k,
                        "value": v,
                        "page": 1,
                        "legibility": "clear" if v else "absent",
                        "handwritten": False,
                        "evidence": f"label {k}",
                        "bbox": [50, 50, 600, 200] if v else None,
                        "ambiguous_characters": [],
                    }
                    for k, v in self.full_values.items()
                ]
            }
        task = parts[-1].text
        spec = next(f for f in DOC_SPECS[self.doc_type].fields if f"'{f.label}'" in task)
        table = self.binarized_values if "CHARACTER BY CHARACTER" in task else self.crop_values
        value = table.get(spec.key, self.full_values.get(spec.key))
        return {
            "value": value,
            "legibility": "clear" if value else "absent",
            "label_visible": True,
            "ambiguous_characters": [],
            "evidence": spec.label,
        }


def _pan_file():
    return (
        "card.png",
        png_bytes(
            make_text_image(
                [
                    "INCOME TAX DEPARTMENT",
                    "Permanent Account Number Card",
                    "ABCPK1234Q",
                    "Name",
                    "TEST PERSON",
                    "Father's Name",
                    "TEST FATHER",
                    "Date of Birth",
                    "01/02/1990",
                ]
            )
        ),
    )


PAN_VALUES = {"pan_number": "ABCPK1234Q", "full_name": "TEST PERSON", "fathers_name": "TEST FATHER", "date_of_birth": "01/02/1990"}


def test_printed_document_fields_are_accepted(fake_llm_factory):
    llm = fake_llm_factory(VisionScript(DocumentType.PAN_CARD, PAN_VALUES))
    batch = process_files([_pan_file()], llm, threshold=0.85)
    doc = batch.documents[0]
    assert doc.classification.document_type == DocumentType.PAN_CARD
    pan = doc.fields["pan_number"]
    assert pan.normalized_value == "ABCPK1234Q" and pan.raw_value == "ABCPK1234Q"
    assert pan.confidence >= 0.85 and not pan.review_required
    assert "vision_full_page" in pan.method and "vision_crop" in pan.method and "validator" in pan.method
    assert doc.fields["date_of_birth"].normalized_value == "1990-02-01"
    assert set(doc.fields) == set(EXPECTED_FIELDS[DocumentType.PAN_CARD])


def test_missing_field_has_zero_confidence_and_review(fake_llm_factory):
    values = dict(PAN_VALUES, fathers_name=None)
    llm = fake_llm_factory(VisionScript(DocumentType.PAN_CARD, values, crop_values={"fathers_name": None}))
    doc = process_files([_pan_file()], llm).documents[0]
    f = doc.fields["fathers_name"]
    assert f.confidence == 0.0 and f.review_required and f.raw_value is None
    assert doc.document_review_required


def test_handwritten_disagreement_withholds_value(fake_llm_factory):
    full = {
        "bank_account_number": "31004258912",
        "ifsc_code": "SBIN0227112",
        "bank_name": "State Bank of India",
        "amount_in_figures": "50,000",
        "frequency": "As & when presented",
        "amount_in_words": "Fifty thousand only",
    }
    llm = fake_llm_factory(
        VisionScript(
            DocumentType.NACH_MANDATE,
            full,
            crop_values={"bank_account_number": "31004258972"},
            binarized_values={"bank_account_number": "31004258912 0"},
        )
    )
    img = png_bytes(make_text_image(["NACH MANDATE INSTRUCTION", "UMRN", "Bank a/c number", "IFSC  or MICR", "FREQUENCY"]))
    doc = process_files([("m.png", img)], llm).documents[0]
    acct = doc.fields["bank_account_number"]
    assert acct.raw_value is None and acct.normalized_value is None
    assert acct.review_required and acct.confidence <= 0.40
    assert len(acct.candidates) == 3
    assert any("disagree" in r for r in acct.review_reasons)
    ifsc = doc.fields["ifsc_code"]
    assert ifsc.normalized_value == "SBIN0227112" and "vision_crop_binarized" in ifsc.method
    assert any("matches amount in words" in w for w in doc.warnings)


def test_validation_failure_is_preserved_and_flagged(fake_llm_factory):
    values = dict(PAN_VALUES, pan_number="ABCDE1234F")
    llm = fake_llm_factory(VisionScript(DocumentType.PAN_CARD, values))
    f = process_files([_pan_file()], llm).documents[0].fields["pan_number"]
    assert f.raw_value == "ABCDE1234F" and f.normalized_value == "ABCDE1234F"
    assert f.review_required and f.confidence <= 0.70
    assert any("holder-type" in r for r in f.review_reasons)


def test_unsupported_multipage_pdf_is_one_unknown_document(fake_llm_factory):
    llm = fake_llm_factory(VisionScript(DocumentType.UNKNOWN, {}, groups=[[1, 2]]))
    data = pdf_bytes([["ASSIGNMENT REQUEST FORM", "Details of Assignor"], ["Endorsement on the Policy document"]])
    batch = process_files([("other.pdf", data)], llm, pdf_dpi=100)
    assert len(batch.documents) == 1
    doc = batch.documents[0]
    assert doc.pages == [1, 2] and doc.classification.document_type == DocumentType.UNKNOWN
    assert doc.document_review_required and doc.fields == {}
    rows = flag_rows(batch.documents)
    assert rows and rows[0]["field_name"] == "(document classification)"


def test_classifier_page_groups_are_sanitised(fake_llm_factory):
    llm = fake_llm_factory(VisionScript(DocumentType.UNKNOWN, {}, groups=[[1, 1, 7]]))
    data = pdf_bytes([["a"], ["b"]])
    docs = process_files([("x.pdf", data)], llm, pdf_dpi=72).documents
    assert sorted(p for d in docs for p in d.pages) == [1, 2]


def test_bad_file_is_reported_not_raised(fake_llm_factory):
    batch = process_files([("bad.png", b"nope")], fake_llm_factory(None))
    assert "bad.png" in batch.file_errors and batch.documents == []


def test_offline_mode_without_llm(pan_like_png):
    doc = process_files([("card.png", pan_like_png)], None).documents[0]
    assert doc.classification.review_required
    assert all(f.confidence == 0.0 and f.review_required for f in doc.fields.values())


def test_json_serialisation_and_reports(tmp_path, fake_llm_factory):
    llm = fake_llm_factory(VisionScript(DocumentType.PAN_CARD, dict(PAN_VALUES, fathers_name=None), crop_values={"fathers_name": None}))
    batch = process_files([_pan_file()], llm)
    paths = write_outputs(batch.documents, tmp_path, 0.85)
    doc_json = json.loads((tmp_path / "json" / f"{batch.documents[0].document_id}.json").read_text())
    for key in ("document_id", "source_file", "pages", "classification", "fields", "document_review_required", "warnings"):
        assert key in doc_json
    field = doc_json["fields"]["pan_number"]
    for key in ("raw_value", "normalized_value", "confidence", "page", "method", "evidence", "review_required", "review_reasons"):
        assert key in field
    DocumentResult.model_validate(doc_json)  # round-trips
    rows = list(csv.DictReader(io.StringIO(paths["flagging_csv"].read_text())))
    assert list(rows[0].keys()) == FLAG_COLUMNS
    assert any(r["field_name"] == "fathers_name" and float(r["threshold"]) == 0.85 for r in rows)
    assert json.loads(paths["flagging_json"].read_text())["flag_count"] == len(rows)
    summary = summary_rows(batch.documents)[0]
    assert summary["fields_missing"] == 1
    assert "threshold_rationale" in json.loads(paths["all_results"].read_text())


def test_threshold_is_configurable(fake_llm_factory):
    llm = fake_llm_factory(VisionScript(DocumentType.PAN_CARD, PAN_VALUES))
    strict = process_files([_pan_file()], llm, threshold=1.0).documents[0]
    assert all(f.review_required for f in strict.fields.values() if f.confidence < 1.0)


def test_csv_helper():
    assert to_csv([{"a": 1, "b": 2}], ["a"]).splitlines() == ["a", "1"]


# --------------------------------------------------------------------------- schemas


def test_classification_schema_lists_every_type():
    enum = CLASSIFY_SCHEMA["properties"]["documents"]["items"]["properties"]["document_type"]["enum"]
    assert set(enum) == {t.value for t in DocumentType}


@pytest.mark.parametrize("doc_type", SUPPORTED_TYPES, ids=lambda t: t.value)
def test_every_supported_document_schema(doc_type):
    spec = DOC_SPECS[doc_type]
    assert [f.key for f in spec.required_fields] == EXPECTED_FIELDS[doc_type]
    schema = _full_page_schema(spec)
    enum = schema["properties"]["fields"]["items"]["properties"]["field"]["enum"]
    assert set(EXPECTED_FIELDS[doc_type]) <= set(enum)
    assert all(f.anchors for f in spec.fields)
    assert spec.keywords
