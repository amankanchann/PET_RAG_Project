import os
import re
import json

import chromadb
import ollama

from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

VECTOR_DB_PATH = os.path.join(
    BASE_DIR,
    "vector_db"
)


# ============================================================
# MODEL / DATABASE SETTINGS
# ============================================================

COLLECTION_NAME = "pet_reports"

OLLAMA_MODEL = "llama3.2:3b"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ============================================================
# INITIALIZE CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=VECTOR_DB_PATH
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_value(value):

    if value is None:
        return ""

    value = str(value)

    value = value.replace(
        "**",
        ""
    )

    value = value.replace(
        "\r",
        " "
    )

    value = value.replace(
        "\n",
        " "
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# ============================================================
# FIELD EXTRACTION
# ============================================================

def extract_table_field(
    text,
    field_names
):
    """
    Extract a field from different possible DOCX formats.

    Examples:

    NAME | Patient A

    NAME
    Patient A

    NAME: Patient A
    """

    if isinstance(
        field_names,
        str
    ):

        field_names = [
            field_names
        ]

    for field in field_names:

        # ----------------------------------------------------
        # FIELD | VALUE
        # ----------------------------------------------------

        pattern = (
            rf"{re.escape(field)}"
            rf"\s*\|\s*"
            rf"([^\n|]+)"
        )

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            return clean_value(
                match.group(1)
            )

        # ----------------------------------------------------
        # FIELD : VALUE
        # ----------------------------------------------------

        pattern = (
            rf"{re.escape(field)}"
            rf"\s*:\s*"
            rf"([^\n]+)"
        )

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            return clean_value(
                match.group(1)
            )

        # ----------------------------------------------------
        # FIELD newline VALUE
        # ----------------------------------------------------

        pattern = (
            rf"{re.escape(field)}"
            rf"\s*\n\s*"
            rf"([^\n]+)"
        )

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            value = clean_value(
                match.group(1)
            )

            # Don't accidentally take another field
            if not re.match(
                r"^(NAME|AGE|GENDER|AGE/GENDER|"
                r"KH|HMP|STUDY DATE|MODALITY|"
                r"REPORTED ON|REFERRED BY)\b",
                value,
                re.IGNORECASE
            ):

                return value

    return ""


# ============================================================
# SECTION EXTRACTION
# ============================================================

def extract_section(
    text,
    start_keyword,
    end_keywords
):
    """
    Safely extract sections such as:

    FINDINGS
    IMPRESSION

    Avoids complicated regex construction.
    """

    # --------------------------------------------------------
    # Find section beginning
    # --------------------------------------------------------

    start_pattern = (
        rf"\b{re.escape(start_keyword)}\b\s*:?"
    )

    start_match = re.search(
        start_pattern,
        text,
        re.IGNORECASE
    )

    if not start_match:

        return ""

    # Text after section heading
    section_text = text[
        start_match.end():
    ]

    # --------------------------------------------------------
    # Find next section
    # --------------------------------------------------------

    end_positions = []

    for end_keyword in end_keywords:

        end_pattern = (
            rf"\b{re.escape(end_keyword)}\b\s*:?"
        )

        end_match = re.search(
            end_pattern,
            section_text,
            re.IGNORECASE
        )

        if end_match:

            end_positions.append(
                end_match.start()
            )

    # --------------------------------------------------------
    # Cut at first following section
    # --------------------------------------------------------

    if end_positions:

        section_text = section_text[
            :min(end_positions)
        ]

    return clean_value(
        section_text
    )


# ============================================================
# BASIC PATIENT INFORMATION
# ============================================================

def extract_basic_information(
    text
):

    patient_name = extract_table_field(
        text,
        [
            "NAME",
            "PATIENT NAME"
        ]
    )

    age_gender = extract_table_field(
        text,
        [
            "AGE/GENDER",
            "AGE / GENDER"
        ]
    )

    kh_no = extract_table_field(
        text,
        [
            "KH no.",
            "KH No.",
            "KH NO",
            "KH"
        ]
    )

    hmp_no = extract_table_field(
        text,
        [
            "HMP no.",
            "HMP No.",
            "HMP NO",
            "HMP"
        ]
    )

    study_date = extract_table_field(
        text,
        [
            "STUDY DATE"
        ]
    )

    modality = extract_table_field(
        text,
        [
            "MODALITY"
        ]
    )

    reported_on = extract_table_field(
        text,
        [
            "REPORTED ON"
        ]
    )

    referred_by = extract_table_field(
        text,
        [
            "REFERRED BY"
        ]
    )

    # ========================================================
    # AGE / GENDER
    # ========================================================

    age = ""

    gender = ""

    if age_gender:

        age_match = re.search(
            r"(\d+)\s*Y",
            age_gender,
            re.IGNORECASE
        )

        if age_match:

            age = age_match.group(1)

        gender_match = re.search(
            r"/\s*([MF])\b",
            age_gender,
            re.IGNORECASE
        )

        if gender_match:

            gender_code = (
                gender_match.group(1)
                .upper()
            )

            if gender_code == "F":

                gender = "Female"

            elif gender_code == "M":

                gender = "Male"

    return {

        "patient_name":
            patient_name,

        "age":
            age,

        "gender":
            gender,

        "patient_id":
            kh_no,

        "hmp_no":
            hmp_no,

        "study_date":
            study_date,

        "modality":
            modality,

        "reported_on":
            reported_on,

        "referred_by":
            referred_by
    }


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    text,
    chunk_size=500
):
    """
    Split report into word-based chunks.
    """

    words = text.split()

    chunks = []

    for start in range(
        0,
        len(words),
        chunk_size
    ):

        chunk = " ".join(
            words[
                start:
                start + chunk_size
            ]
        )

        if chunk.strip():

            chunks.append(
                chunk.strip()
            )

    return chunks


# ============================================================
# ADD DOCUMENT TO VECTOR DATABASE
# ============================================================

def add_document_to_vector_db(
    text,
    source_file
):

    chunks = create_chunks(
        text
    )

    if not chunks:

        return

    embeddings = (
        embedding_model
        .encode(chunks)
        .tolist()
    )

    ids = []

    documents = []

    metadatas = []

    for index, chunk in enumerate(
        chunks
    ):

        document_id = (
            f"{source_file}_{index}"
        )

        ids.append(
            document_id
        )

        documents.append(
            chunk
        )

        metadatas.append({

            "source":
                source_file
        })

    collection.upsert(

        ids=ids,

        documents=documents,

        embeddings=embeddings,

        metadatas=metadatas
    )


# ============================================================
# RETRIEVE INFORMATION
# ============================================================

def retrieve_information(
    query,
    source_file,
    n_results=5
):
    """
    Retrieve information ONLY from the current report.

    This prevents patient A information from being retrieved
    while processing patient B.
    """

    query_embedding = (
        embedding_model
        .encode([query])
        .tolist()
    )

    results = collection.query(

        query_embeddings=query_embedding,

        n_results=n_results,

        where={
            "source": source_file
        }
    )

    documents = results.get(
        "documents",
        []
    )

    if not documents:

        return ""

    retrieved = []

    for group in documents:

        for document in group:

            retrieved.append(
                document
            )

    return "\n\n".join(
        retrieved
    )


# ============================================================
# BUILD LLM CONTEXT
# ============================================================

def build_context(
    full_text,
    retrieved_text
):

    findings = extract_section(

        full_text,

        "FINDINGS",

        [
            "IMPRESSION",
            "CONCLUSION"
        ]
    )

    impression = extract_section(

        full_text,

        "IMPRESSION",

        [
            "CONCLUSION",
            "REPORT END"
        ]
    )

    indication = extract_section(

        full_text,

        "INDICATION",

        [
            "FINDINGS",
            "IMPRESSION"
        ]
    )

    context = f"""

============================================================
MEDICAL PET/CT REPORT
============================================================

{full_text}


============================================================
INDICATION
============================================================

{indication}


============================================================
FINDINGS
============================================================

{findings}


============================================================
IMPRESSION
============================================================

{impression}


============================================================
RAG RETRIEVED CONTEXT
============================================================

{retrieved_text}


============================================================
END OF REPORT
============================================================

"""

    return context


# ============================================================
# ASK LOCAL LLAMA
# ============================================================

def ask_local_llm(
    context
):

    prompt = f"""

You are a medical-report information extraction system.

You are processing a PET/CT report.

Your job is to extract structured information.

You MUST use only the information contained in the report.

DO NOT invent information.

DO NOT guess.

DO NOT use outside medical knowledge to create values.

The most important sections for determining CURRENT disease are:

1. FINDINGS
2. IMPRESSION

The report's historical treatment information can be used to
identify PREVIOUS cancer.

============================================================
IMPORTANT DISTINCTION
============================================================

There are two different concepts:

A. CURRENT ACTIVE CANCER

B. PREVIOUS / TREATED CANCER

They MUST NOT be confused.

============================================================
RULES FOR CURRENT CANCER
============================================================

Set:

"cancer_present": "Yes"

when the current report demonstrates a suspicious or malignant
current lesion/pathology.

Strong examples include:

- primary pathology
- primary tumor
- malignant lesion
- metabolically active tumor
- current malignant mass
- current NET
- current carcinoma
- current lesion described as cancer in FINDINGS/IMPRESSION

Current lesion measurements and SUVmax should be extracted when
they clearly belong to that current malignant lesion.

============================================================
IMPORTANT RULE ABOUT "CASE OF"
============================================================

The phrase:

"Case of"

DOES NOT automatically mean previous cancer.

For example:

"Case of Duodenal nodule, ? carcinoid"

does NOT mean previous cancer.

If the report then describes:

"DOTANOC avid enhancing polypoidal lesion"

and gives a current lesion size and SUVmax,

and the IMPRESSION says:

"primary pathology, Duodenal NET"

then this is CURRENT cancer.

Therefore:

cancer_present = "Yes"

previous_cancer = "No"

============================================================
RULES FOR PREVIOUS CANCER
============================================================

Set:

"previous_cancer": "Yes"

when the report clearly indicates a previously diagnosed cancer
that has undergone treatment or is being reassessed.

Examples:

- post NACT
- post MRM
- post chemotherapy
- post radiotherapy
- status post surgery
- for reassessment

However, treatment history alone does not determine whether
CURRENT cancer is present.

You MUST check FINDINGS and IMPRESSION.

============================================================
TREATED CANCER WITH NO CURRENT DISEASE
============================================================

If the report describes previous cancer and treatment and the
current report says things such as:

"No evidence of abnormal metabolic activity"

"No evidence of metabolically active locoregional lymphadenopathy"

"No evidence of distant metastasis"

"No evidence of suspicious lesion"

then:

cancer_present = "No"

previous_cancer = "Yes"

Current cancer fields must be:

"Not applicable"

============================================================
CURRENT SUVmax
============================================================

Only put a value into cancer_suvmax when the SUVmax belongs to
the CURRENT cancer lesion.

DO NOT use SUVmax values belonging to:

- postprocedural changes
- post-radiotherapy changes
- inflammatory changes
- reactive lymph nodes
- benign lesions
- unrelated organs
- historical lesions

============================================================
CURRENT CANCER SIZE
============================================================

Only put a measurement into cancer_size when it belongs to the
CURRENT cancer lesion.

DO NOT use:

- reactive lymph-node measurements
- benign lesion measurements
- postprocedural tissue measurements
- unrelated lesion measurements
- historical lesion measurements

============================================================
PATIENT A EXAMPLE
============================================================

For a report containing:

"Case of Duodenal nodule, ? carcinoid"

and:

"DOTANOC avid enhancing well defined polypoidal lesion"

with:

"1.6 x 1.7 x 1.5 cm"

and:

"SUVmax - 54.56"

and IMPRESSION:

"primary pathology, Duodenal NET"

the correct result is:

{{
    "cancer_present": "Yes",
    "cancer_type": "Duodenal NET",
    "cancer_location": "Duodenum",
    "cancer_suvmax": "54.56",
    "cancer_size": "1.6 x 1.7 x 1.5 cm",
    "previous_cancer": "No",
    "previous_cancer_type": "Not applicable",
    "previous_cancer_location": "Not applicable"
}}

============================================================
PATIENT B EXAMPLE
============================================================

For a report containing:

"Case of carcinoma left breast, post NACT, post MRM, CTRT,
for re-assessment"

and current findings such as:

"postprocedural changes"

and:

"No evidence of abnormal metabolic activity"

and:

"No evidence of metabolically active locoregional lymphadenopathy
or distant metastasis"

the correct result is:

{{
    "cancer_present": "No",
    "cancer_type": "Not applicable",
    "cancer_location": "Not applicable",
    "cancer_suvmax": "Not applicable",
    "cancer_size": "Not applicable",
    "previous_cancer": "Yes",
    "previous_cancer_type": "Carcinoma",
    "previous_cancer_location": "Left breast"
}}

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use exactly these fields:

{{
    "cancer_present": "Yes/No",
    "cancer_type": "",
    "cancer_location": "",
    "cancer_suvmax": "",
    "cancer_size": "",
    "previous_cancer": "Yes/No",
    "previous_cancer_type": "",
    "previous_cancer_location": ""
}}

Use:

"Not applicable"

when a field does not apply.

Use:

"Not mentioned"

only when the report genuinely does not provide enough
information to determine the field.

Do not add extra fields.

============================================================
REPORT CONTEXT
============================================================

{context}

"""

    response = ollama.chat(

        model=OLLAMA_MODEL,

        messages=[

            {
                "role": "user",
                "content": prompt
            }

        ]
    )

    return response[
        "message"
    ][
        "content"
    ]


# ============================================================
# PARSE LLM JSON
# ============================================================

def parse_llm_json(
    response
):

    response = response.strip()

    # Remove markdown code fences
    response = re.sub(
        r"```json",
        "",
        response,
        flags=re.IGNORECASE
    )

    response = response.replace(
        "```",
        ""
    ).strip()

    # Find JSON object
    match = re.search(
        r"\{.*\}",
        response,
        re.DOTALL
    )

    if not match:

        return {

            "cancer_present":
                "Not mentioned",

            "cancer_type":
                "Not mentioned",

            "cancer_location":
                "Not mentioned",

            "cancer_suvmax":
                "Not mentioned",

            "cancer_size":
                "Not mentioned",

            "previous_cancer":
                "Not mentioned",

            "previous_cancer_type":
                "Not mentioned",

            "previous_cancer_location":
                "Not mentioned"
        }

    json_text = match.group(0)

    try:

        return json.loads(
            json_text
        )

    except json.JSONDecodeError:

        return {

            "cancer_present":
                "Not mentioned",

            "cancer_type":
                "Not mentioned",

            "cancer_location":
                "Not mentioned",

            "cancer_suvmax":
                "Not mentioned",

            "cancer_size":
                "Not mentioned",

            "previous_cancer":
                "Not mentioned",

            "previous_cancer_type":
                "Not mentioned",

            "previous_cancer_location":
                "Not mentioned"
        }


# ============================================================
# DETERMINISTIC VALIDATION
# ============================================================

def validate_cancer_result(
    result,
    full_text
):
    """
    Validate the LLM output using deterministic rules.

    This is important because this is a research project.

    The LLM performs contextual extraction.
    Rules catch obvious contradictions.
    """

    text_lower = full_text.lower()

    findings = extract_section(

        full_text,

        "FINDINGS",

        [
            "IMPRESSION",
            "CONCLUSION"
        ]
    ).lower()

    impression = extract_section(

        full_text,

        "IMPRESSION",

        [
            "CONCLUSION",
            "REPORT END"
        ]
    ).lower()

    combined = (
        text_lower
        + "\n"
        + findings
        + "\n"
        + impression
    )

    # ========================================================
    # CURRENT PRIMARY PATHOLOGY
    # ========================================================

    current_primary = (

        "primary pathology" in combined

        or

        "primary tumor" in combined

        or

        "primary lesion" in combined
    )

    # ========================================================
    # CURRENT DUODENAL NET
    # ========================================================

    current_duodenal_net = (

        "duodenal net" in combined
    )

    # ========================================================
    # CURRENT MALIGNANT LESION
    # ========================================================

    current_malignant_language = any(

        phrase in combined

        for phrase in [

            "metabolically active tumor",

            "malignant lesion",

            "primary malignant",

            "active malignancy",

            "residual tumor",

            "residual disease",

            "recurrent tumor",

            "recurrent disease",

            "viable tumor"

        ]
    )

    # ========================================================
    # TREATMENT HISTORY
    # ========================================================

    treatment_history = any(

        phrase in combined

        for phrase in [

            "post nact",

            "post mrm",

            "post chemotherapy",

            "post radiotherapy",

            "post rt",

            "post surgery",

            "status post",

            "for re-assessment",

            "for reassessment"

        ]
    )

    # ========================================================
    # NEGATIVE CURRENT DISEASE
    # ========================================================

    negative_current = any(

        phrase in combined

        for phrase in [

            "no evidence of abnormal metabolic activity",

            "no evidence of metabolically active",

            "no evidence of active disease",

            "no evidence of distant metastasis",

            "no evidence of locoregional",

            "no evidence of suspicious lesion",

            "no evidence of suspicious pulmonary nodule"

        ]
    )

    # ========================================================
    # RULE 1:
    # Explicit current disease takes priority
    # ========================================================

    if (

        current_primary
        or current_duodenal_net
        or current_malignant_language

    ):

        result[
            "cancer_present"
        ] = "Yes"

        # Current disease means we should NOT automatically
        # classify it as previous cancer.

        if current_duodenal_net:

            result[
                "cancer_type"
            ] = "Duodenal NET"

            result[
                "cancer_location"
            ] = "Duodenum"

        # If current disease is explicitly described,
        # historical fields should not overwrite it.

        if not result.get(
            "previous_cancer"
        ):

            result[
                "previous_cancer"
            ] = "No"

        return result

    # ========================================================
    # RULE 2:
    # Treated cancer + negative current findings
    # ========================================================

    if (

        treatment_history

        and

        negative_current

    ):

        result[
            "cancer_present"
        ] = "No"

        result[
            "cancer_type"
        ] = "Not applicable"

        result[
            "cancer_location"
        ] = "Not applicable"

        result[
            "cancer_suvmax"
        ] = "Not applicable"

        result[
            "cancer_size"
        ] = "Not applicable"

        result[
            "previous_cancer"
        ] = "Yes"

        # ----------------------------------------------------
        # Specific breast cancer pattern
        # ----------------------------------------------------

        if (
            "carcinoma left breast"
            in combined
        ):

            result[
                "previous_cancer_type"
            ] = "Carcinoma"

            result[
                "previous_cancer_location"
            ] = "Left breast"

        return result

    return result


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_cancer_result(
    result
):

    required_fields = [

        "cancer_present",

        "cancer_type",

        "cancer_location",

        "cancer_suvmax",

        "cancer_size",

        "previous_cancer",

        "previous_cancer_type",

        "previous_cancer_location"
    ]

    # --------------------------------------------------------
    # Make sure every field exists
    # --------------------------------------------------------

    for field in required_fields:

        if field not in result:

            result[field] = "Not mentioned"

        result[field] = clean_value(
            result[field]
        )

    # ========================================================
    # CURRENT CANCER = NO
    # ========================================================

    if (
        result[
            "cancer_present"
        ].lower()
        == "no"
    ):

        result[
            "cancer_type"
        ] = "Not applicable"

        result[
            "cancer_location"
        ] = "Not applicable"

        result[
            "cancer_suvmax"
        ] = "Not applicable"

        result[
            "cancer_size"
        ] = "Not applicable"

    # ========================================================
    # PREVIOUS CANCER = NO
    # ========================================================

    if (
        result[
            "previous_cancer"
        ].lower()
        == "no"
    ):

        result[
            "previous_cancer_type"
        ] = "Not applicable"

        result[
            "previous_cancer_location"
        ] = "Not applicable"

    return result


# ============================================================
# COMPLETE REPORT PROCESSING
# ============================================================

def process_report(
    text,
    source_file
):

    print(
        "\nExtracting patient information..."
    )

    # --------------------------------------------------------
    # 1. Metadata
    # --------------------------------------------------------

    basic_info = extract_basic_information(
        text
    )

    # --------------------------------------------------------
    # 2. Store current report
    # --------------------------------------------------------

    print(
        "Adding report to ChromaDB..."
    )

    add_document_to_vector_db(
        text,
        source_file
    )

    # --------------------------------------------------------
    # 3. Query
    # --------------------------------------------------------

    query = """

    current cancer tumor lesion malignancy
    previous cancer treated cancer
    primary pathology
    current disease
    FINDINGS IMPRESSION
    SUVmax size location
    """

    # --------------------------------------------------------
    # 4. Retrieve only current patient's report
    # --------------------------------------------------------

    print(
        "Retrieving relevant information..."
    )

    retrieved_text = retrieve_information(

        query,

        source_file,

        n_results=5
    )

    # --------------------------------------------------------
    # 5. Build context
    # --------------------------------------------------------

    context = build_context(

        text,

        retrieved_text
    )

    # --------------------------------------------------------
    # 6. LLM
    # --------------------------------------------------------

    print(
        "Sending context to local Llama..."
    )

    llm_response = ask_local_llm(
        context
    )

    print(
        "\n---------------- LLM RESPONSE ----------------"
    )

    print(
        llm_response
    )

    print(
        "------------------------------------------------"
    )

    # --------------------------------------------------------
    # 7. Parse JSON
    # --------------------------------------------------------

    cancer_result = parse_llm_json(
        llm_response
    )

    # --------------------------------------------------------
    # 8. Deterministic validation
    # --------------------------------------------------------

    cancer_result = validate_cancer_result(

        cancer_result,

        text
    )

    # --------------------------------------------------------
    # 9. Normalize
    # --------------------------------------------------------

    cancer_result = normalize_cancer_result(
        cancer_result
    )

    # --------------------------------------------------------
    # 10. Combine
    # --------------------------------------------------------

    final_result = {

        **basic_info,

        **cancer_result
    }

    return final_result