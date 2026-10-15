import os
import sys

from document_loader import (
    extract_text_from_docx
)

from rag_pipeline import (
    process_report
)

from excel_export import (
    export_to_excel
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# FOLDERS
# ============================================================

REPORTS_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)


# ============================================================
# OUTPUT FILE
# ============================================================

OUTPUT_FILE = os.path.join(

    OUTPUT_DIR,

    "PET_Cancer_Extraction.xlsx"
)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "PET/CT MEDICAL REPORT RAG EXTRACTION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nReports folder:\n{REPORTS_DIR}"
    )

    # --------------------------------------------------------
    # Check reports directory
    # --------------------------------------------------------

    if not os.path.exists(
        REPORTS_DIR
    ):

        print(
            "\nERROR: reports folder does not exist."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Find DOCX files
    # --------------------------------------------------------

    files = [

        file

        for file in os.listdir(
            REPORTS_DIR
        )

        if file.lower().endswith(
            ".docx"
        )
    ]

    if not files:

        print(
            "\nERROR: No DOCX files found."
        )

        sys.exit(1)

    print(
        f"\nFound {len(files)} DOCX report(s)."
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results = []

    # ========================================================
    # PROCESS EACH REPORT
    # ========================================================

    for file in sorted(files):

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"Processing: {file}"
        )

        print(
            "-" * 70
        )

        file_path = os.path.join(

            REPORTS_DIR,

            file
        )

        try:

            # ------------------------------------------------
            # Extract DOCX
            # ------------------------------------------------

            print(
                "\n1. Extracting DOCX text..."
            )

            text = extract_text_from_docx(
                file_path
            )

            if not text.strip():

                print(
                    "ERROR: No text extracted."
                )

                continue

            print(
                f"Extracted {len(text)} characters."
            )

            # ------------------------------------------------
            # Process through RAG pipeline
            # ------------------------------------------------

            print(
                "\n2. Running RAG extraction..."
            )

            result = process_report(

                text,

                file
            )

            # ------------------------------------------------
            # Add source filename
            # ------------------------------------------------

            result[
                "source_file"
            ] = file

            results.append(
                result
            )

            # ------------------------------------------------
            # Display result
            # ------------------------------------------------

            print(
                "\n3. FINAL EXTRACTED INFORMATION"
            )

            print(
                "-" * 50
            )

            for key, value in result.items():

                print(
                    f"{key}: {value}"
                )

        except Exception as error:

            print(
                f"\nERROR processing {file}:"
            )

            print(
                str(error)
            )

    # ========================================================
    # EXPORT
    # ========================================================

    if not results:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "NO REPORTS WERE SUCCESSFULLY PROCESSED"
        )

        print(
            "=" * 70
        )

        return

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Export Excel
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Creating Excel file..."
    )

    print(
        "=" * 70
    )

    df = export_to_excel(

        results,

        OUTPUT_FILE
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print(
        "\nExcel successfully created:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nFinal table:"
    )

    print(
        df.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PROCESS COMPLETED"
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()