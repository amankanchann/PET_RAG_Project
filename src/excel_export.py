import os

import pandas as pd


def export_to_excel(
    results,
    output_file
):
    """
    Convert extracted report results into an Excel file.
    """

    rows = []

    for result in results:

        row = {

            "Patient Name":
                result.get(
                    "patient_name",
                    ""
                ),

            "Age":
                result.get(
                    "age",
                    ""
                ),

            "Gender":
                result.get(
                    "gender",
                    ""
                ),

            "KH No.":
                result.get(
                    "patient_id",
                    ""
                ),

            "HMP No.":
                result.get(
                    "hmp_no",
                    ""
                ),

            "Study Date":
                result.get(
                    "study_date",
                    ""
                ),

            "Modality":
                result.get(
                    "modality",
                    ""
                ),

            "Reported On":
                result.get(
                    "reported_on",
                    ""
                ),

            "Referred By":
                result.get(
                    "referred_by",
                    ""
                ),

            "Cancer Present":
                result.get(
                    "cancer_present",
                    ""
                ),

            "Cancer Type":
                result.get(
                    "cancer_type",
                    ""
                ),

            "Cancer Location":
                result.get(
                    "cancer_location",
                    ""
                ),

            "Cancer SUVmax":
                result.get(
                    "cancer_suvmax",
                    ""
                ),

            "Cancer Size":
                result.get(
                    "cancer_size",
                    ""
                ),

            "Previous Cancer":
                result.get(
                    "previous_cancer",
                    ""
                ),

            "Previous Cancer Type":
                result.get(
                    "previous_cancer_type",
                    ""
                ),

            "Previous Cancer Location":
                result.get(
                    "previous_cancer_location",
                    ""
                ),

            "Source File":
                result.get(
                    "source_file",
                    ""
                )
        }

        rows.append(
            row
        )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_directory = os.path.dirname(
        output_file
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_excel(
        output_file,
        index=False
    )

    return df