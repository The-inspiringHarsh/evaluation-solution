"""Document types, field specifications and the output data model (Pydantic)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    AADHAAR_CARD = "aadhaar_card"
    PAN_CARD = "pan_card"
    DRIVING_LICENCE = "driving_licence"
    PASSPORT = "passport"
    NACH_MANDATE = "nach_ecs_mandate"
    FATCA_ANNEXURE = "fatca_annexure"
    BENEFIT_ILLUSTRATION = "benefit_illustration_declaration"
    MORAL_HAZARD = "moral_hazard_questionnaire"
    MULTIPLE_POLICIES = "multiple_policies_consent"
    SUITABILITY_PROFILER = "suitability_profiler_declaration"
    UNKNOWN = "unknown_or_other"


Region = Literal["right", "below", "right_wide", "row"]
Kind = Literal["printed", "handwritten", "checkbox"]


@dataclass(frozen=True)
class FieldSpec:
    """How to find, read and validate one target field."""

    key: str
    label: str
    description: str
    kind: Kind
    validator: Optional[str] = None
    anchors: tuple[str, ...] = ()
    region: Region = "right"
    options: tuple[str, ...] = ()  # for checkbox fields
    auxiliary: bool = False  # extracted only to support validation (not a required output field)


@dataclass(frozen=True)
class DocSpec:
    doc_type: DocumentType
    display: str
    cues: str  # what the document looks like (for the classifier prompt)
    keywords: tuple[str, ...]  # printed phrases used by the independent OCR keyword classifier
    fields: tuple[FieldSpec, ...] = field(default_factory=tuple)

    @property
    def required_fields(self) -> tuple[FieldSpec, ...]:
        return tuple(f for f in self.fields if not f.auxiliary)


DOC_SPECS: dict[DocumentType, DocSpec] = {
    DocumentType.AADHAAR_CARD: DocSpec(
        DocumentType.AADHAAR_CARD,
        "Aadhaar Card",
        "UIDAI identity card with a 12-digit Aadhaar number, name, DOB, gender, address.",
        ("aadhaar", "unique identification authority", "uidai", "enrollment", "enrolment", "mera aadhaar"),
        (
            FieldSpec(
                "aadhaar_number",
                "Aadhaar Number",
                "12-digit number, usually printed as 3 groups of 4",
                "printed",
                "aadhaar",
                ("Aadhaar Number",),
                "right",
            ),
            FieldSpec("full_name", "Full Name", "card holder's name as printed", "printed", "name", ("Name",), "below"),
            FieldSpec(
                "date_of_birth", "Date of Birth", "DOB as printed (DD/MM/YYYY)", "printed", "date_past", ("Date of Birth", "DOB"), "below"
            ),
            FieldSpec("address", "Address", "full postal address block including PIN code", "printed", "address", ("Address",), "below"),
        ),
    ),
    DocumentType.PAN_CARD: DocSpec(
        DocumentType.PAN_CARD,
        "PAN Card",
        "Income Tax Department Permanent Account Number card.",
        ("permanent account number", "income tax department", "income tax"),
        (
            FieldSpec(
                "pan_number",
                "PAN Number",
                "10-character PAN (5 letters, 4 digits, 1 letter)",
                "printed",
                "pan",
                ("Permanent Account Number",),
                "below",
            ),
            FieldSpec("full_name", "Full Name", "card holder's name", "printed", "name", ("Name",), "below"),
            FieldSpec(
                "fathers_name",
                "Father's Name",
                "father's name exactly as printed",
                "printed",
                "name",
                ("Father's Name", "Fathers Name"),
                "below",
            ),
            FieldSpec("date_of_birth", "Date of Birth", "DOB as printed", "printed", "date_past", ("Date of Birth",), "below"),
        ),
    ),
    DocumentType.DRIVING_LICENCE: DocSpec(
        DocumentType.DRIVING_LICENCE,
        "Driving Licence",
        "Indian driving licence card with DL number, name, issue date and validity.",
        ("driving licence", "driving license", "licencing authority", "classes of vehicles", "valid till"),
        (
            FieldSpec(
                "dl_number", "DL Number", "driving licence number (state code + RTO + year + serial)", "printed", "dl", ("DL No",), "right"
            ),
            FieldSpec("name", "Name", "licence holder's name", "printed", "name", ("Name",), "right"),
            FieldSpec("date_of_issue", "Date of Issue", "issue date (DOI)", "printed", "date_past", ("Date of Issue", "DOI"), "right"),
            FieldSpec(
                "valid_till",
                "Valid Till",
                "validity/expiry date of the licence (non-transport)",
                "printed",
                "date",
                ("Valid Till",),
                "right",
            ),
        ),
    ),
    DocumentType.PASSPORT: DocSpec(
        DocumentType.PASSPORT,
        "Passport",
        "Passport data page with photo, personal details and a two-line machine readable zone (MRZ).",
        ("passport", "republic of india", "p<ind", "date of expiry", "place of issue", "country code"),
        (
            FieldSpec(
                "passport_number",
                "Passport Number",
                "passport number (1 letter + 7 digits for India)",
                "printed",
                "passport",
                ("Passport No",),
                "below",
            ),
            FieldSpec(
                "date_of_birth", "Date of Birth", "DOB as printed in the visual zone", "printed", "date_past", ("Date of Birth",), "below"
            ),
            FieldSpec(
                "date_of_expiry", "Date of Expiry", "expiry date in the visual zone", "printed", "date", ("Date of Expiry",), "below"
            ),
            FieldSpec(
                "mrz_line_2",
                "MRZ Line 2",
                "the full SECOND line of the MRZ, 44 characters, including every '<' filler",
                "printed",
                "mrz2",
                ("<<<",),
                "row",
            ),
        ),
    ),
    DocumentType.NACH_MANDATE: DocSpec(
        DocumentType.NACH_MANDATE,
        "NACH/ECS Mandate",
        "Bank debit mandate form (NACH/ECS/auto-debit) with account, IFSC, amount and frequency.",
        ("nach", "mandate", "umrn", "ifsc", "micr", "sponsor bank", "ecs", "debit"),
        (
            FieldSpec(
                "bank_account_number",
                "Bank Account Number",
                "handwritten digits in the 'Bank a/c number' boxes",
                "handwritten",
                "account",
                ("Bank a/c number", "a/c number"),
                "right_wide",
            ),
            FieldSpec(
                "ifsc_code", "IFSC Code", "11-character IFSC written in the boxes after 'IFSC'", "handwritten", "ifsc", ("IFSC",), "right"
            ),
            FieldSpec("bank_name", "Bank Name", "bank name written after 'with bank'", "handwritten", "text", ("with bank",), "right_wide"),
            FieldSpec(
                "amount_in_figures",
                "Amount in figures",
                "amount in figures written in the rupee (₹) box",
                "handwritten",
                "amount",
                ("₹", "or MICR"),
                "right",
            ),
            FieldSpec(
                "frequency",
                "Frequency",
                "which FREQUENCY checkbox is ticked",
                "checkbox",
                "checkbox",
                ("FREQUENCY",),
                "row",
                options=("Monthly", "Quarterly", "Half-Yearly", "Yearly", "As & when presented"),
            ),
            FieldSpec(
                "amount_in_words",
                "Amount in words",
                "amount in words after 'an amount of Rupees'",
                "handwritten",
                "text",
                ("amount of Rupees",),
                "right_wide",
                auxiliary=True,
            ),
        ),
    ),
    DocumentType.FATCA_ANNEXURE: DocSpec(
        DocumentType.FATCA_ANNEXURE,
        "FATCA Annexure Form",
        "Annexure form for reporting under section 285BA (tax residency, TIN, place of birth, nationality).",
        ("annexure", "285ba", "tax residency", "tax identification number", "fatca", "country of birth"),
        (
            FieldSpec(
                "policy_number",
                "Policy Number",
                "handwritten policy number after 'Policy No'",
                "handwritten",
                "digits",
                ("Policy No",),
                "right",
            ),
            FieldSpec(
                "tin_pan",
                "TIN/PAN",
                "handwritten value in the 'Tax Identification Number (TIN)' column of the tax-residency table",
                "handwritten",
                "pan",
                ("Tax Identification Number", "Functional equivalent"),
                "below",
            ),
            FieldSpec(
                "fathers_name",
                "Father's Name",
                "handwritten father's name in Section 3",
                "handwritten",
                "name",
                ("Father's Name",),
                "right",
            ),
            FieldSpec(
                "place_of_birth",
                "Place of Birth",
                "handwritten place of birth in Section 3",
                "handwritten",
                "place",
                ("Place of birth",),
                "right",
            ),
            FieldSpec(
                "nationality", "Nationality", "handwritten nationality in Section 3", "handwritten", "text", ("Nationality",), "right"
            ),
        ),
    ),
    DocumentType.BENEFIT_ILLUSTRATION: DocSpec(
        DocumentType.BENEFIT_ILLUSTRATION,
        "Benefit Illustration Declaration",
        "Customer declaration acknowledging the benefit illustration of a life insurance policy.",
        ("benefit illustration", "customer declaration", "assumed future investment returns"),
        (
            FieldSpec(
                "application_number",
                "Application Number",
                "handwritten Application/Proposal Form Number",
                "handwritten",
                "digits",
                ("Proposal Form Number", "Application/"),
                "right",
            ),
            FieldSpec(
                "policyholder_name",
                "Policyholder Name",
                "handwritten name after 'Name of Policyholder' near the signature",
                "handwritten",
                "name",
                ("Name of Policyhoder", "Name of Policyholder"),
                "right",
            ),
            FieldSpec("date", "Date", "handwritten date near the signature", "handwritten", "date_form", ("Date",), "right"),
            FieldSpec("place", "Place", "handwritten place near the signature", "handwritten", "place", ("Place",), "right"),
        ),
    ),
    DocumentType.MORAL_HAZARD: DocSpec(
        DocumentType.MORAL_HAZARD,
        "Moral Hazard Questionnaire",
        "Questionnaire about nominee/dependents for a life insurance application.",
        ("moral hazard", "questionnaire", "nominee", "dependent"),
        (
            FieldSpec(
                "application_number",
                "Application Number",
                "handwritten value in the 'Application No.' row",
                "handwritten",
                "digits",
                ("Application No",),
                "right",
            ),
            FieldSpec(
                "name_of_life_assured",
                "Name of Life Assured",
                "handwritten value in the 'Name of the Life to be Assured' row",
                "handwritten",
                "name",
                ("Name of the Life to be Assured",),
                "right",
            ),
            FieldSpec(
                "nominee_relationship",
                "Nominee Relationship",
                "handwritten exact relationship of the nominee (question 5)",
                "handwritten",
                "text",
                ("exact relationship",),
                "right",
            ),
            FieldSpec(
                "date", "Date", "handwritten date in the declaration of life to be assured", "handwritten", "date_form", ("Date",), "right"
            ),
            FieldSpec(
                "place", "Place", "handwritten place in the declaration of life to be assured", "handwritten", "place", ("Place",), "right"
            ),
        ),
    ),
    DocumentType.MULTIPLE_POLICIES: DocSpec(
        DocumentType.MULTIPLE_POLICIES,
        "Multiple Policies Consent Form",
        "Customer consent form for split & multiple policies with reasons as checkboxes.",
        ("multiple policies", "split", "consent form", "reason for buying"),
        (
            FieldSpec(
                "proposer_name",
                "Proposer Name",
                "handwritten name after 'Proposer / Life Assured Name'",
                "handwritten",
                "name",
                ("Life Assured Name", "Proposer"),
                "right",
            ),
            FieldSpec(
                "reason_for_multiple_policies",
                "Reason for multiple policies",
                "which reason checkbox is ticked",
                "checkbox",
                "checkbox",
                ("Reason for buying multiple policies",),
                "below",
                options=(
                    "Financial Planning",
                    "Different investment / protection needs",
                    "Separate policy purchased for different beneficiary",
                    "Others",
                ),
            ),
            FieldSpec("date", "Date", "handwritten date at the bottom", "handwritten", "date_form", ("Date",), "right"),
            FieldSpec("place", "Place", "handwritten place at the bottom", "handwritten", "place", ("Place",), "right"),
        ),
    ),
    DocumentType.SUITABILITY_PROFILER: DocSpec(
        DocumentType.SUITABILITY_PROFILER,
        "Suitability Profiler Declaration",
        "Customer declaration that the recommended product suits needs (suitability profiler), signed by life assured and agent.",
        ("suitability profiler", "suitability", "agent/sp", "product recommendation"),
        (
            FieldSpec(
                "application_number",
                "Application Number",
                "handwritten Application/Proposal Form Number",
                "handwritten",
                "digits",
                ("Proposal Form Number",),
                "right",
            ),
            FieldSpec(
                "name_of_life_assured",
                "Name of Life Assured",
                "handwritten name after 'Name of Life Assured' (signature section)",
                "handwritten",
                "name",
                ("Name of Life Assured",),
                "right",
            ),
            FieldSpec(
                "name_of_agent_sp",
                "Name of Agent/SP",
                "handwritten name after 'Name of Agent/SP'",
                "handwritten",
                "name",
                ("Name of Agent/SP",),
                "right",
            ),
            FieldSpec(
                "date",
                "Date",
                "handwritten date next to the life assured's signature (first Date)",
                "handwritten",
                "date_form",
                ("Date",),
                "right",
            ),
            FieldSpec(
                "place",
                "Place",
                "handwritten place next to the life assured's signature (first Place)",
                "handwritten",
                "place",
                ("Place",),
                "right",
            ),
        ),
    ),
}

SUPPORTED_TYPES = [t for t in DocumentType if t is not DocumentType.UNKNOWN]


# --------------------------------------------------------------------------- output model


class FieldResult(BaseModel):
    raw_value: Optional[str] = None
    normalized_value: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    page: Optional[int] = None
    method: list[str] = Field(default_factory=list)
    evidence: str = ""
    review_required: bool = True
    review_reasons: list[str] = Field(default_factory=list)
    handwritten: bool = False
    candidates: list[str] = Field(default_factory=list)
    confidence_components: dict[str, Optional[float]] = Field(default_factory=dict)
    region: Optional[list[int]] = None  # crop box in page pixels [x0, y0, x1, y1]


class Classification(BaseModel):
    document_type: DocumentType
    confidence: float = Field(ge=0.0, le=1.0)
    review_required: bool
    evidence: str = ""
    model_confidence: Optional[float] = None
    keyword_vote: Optional[str] = None


class DocumentResult(BaseModel):
    document_id: str
    source_file: str
    pages: list[int]
    classification: Classification
    fields: dict[str, FieldResult] = Field(default_factory=dict)
    auxiliary_fields: dict[str, FieldResult] = Field(default_factory=dict)
    document_review_required: bool = True
    warnings: list[str] = Field(default_factory=list)
    preprocessing: list[dict] = Field(default_factory=list)
    threshold: float = 0.85
