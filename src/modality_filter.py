import re


def detect_modality(text):
    """
    Detect PET/CT modality from the report.

    This function DOES NOT filter reports.
    It only identifies the modality.
    """

    # ---------------------------------------------------------
    # Explicit MODALITY field
    # ---------------------------------------------------------

    match = re.search(
        r"MODALITY\s*:?\s*\n?\s*\**([^\n*|]+)",
        text,
        re.IGNORECASE
    )

    if match:

        modality = match.group(1).strip()

        modality = modality.replace(
            "**",
            ""
        ).strip()

        # Ga68 DOTA
        if re.search(
            r"Ga[-\s]?68.*DOTA",
            modality,
            re.IGNORECASE
        ):

            return "Ga68 DOTA PET/CT"

        # F-18 FDG
        if re.search(
            r"F[-\s]?18.*FDG",
            modality,
            re.IGNORECASE
        ):

            return "F-18 FDG PET/CT"

        return modality

    # ---------------------------------------------------------
    # Fallback: search complete document
    # ---------------------------------------------------------

    if re.search(
        r"F[-\s]?18\s*FDG\s*PET[/\s]?CT",
        text,
        re.IGNORECASE
    ):

        return "F-18 FDG PET/CT"

    if re.search(
        r"Ga[-\s]?68\s*DOTA\s*PET[/\s]?CT",
        text,
        re.IGNORECASE
    ):

        return "Ga68 DOTA PET/CT"

    return "Unknown"