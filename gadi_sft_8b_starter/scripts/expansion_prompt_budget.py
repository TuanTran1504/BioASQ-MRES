"""Pure expansion-prompt transformations, safe to import during training."""


def five_prompt(ten):
    if ten.count("up to TEN\nanswers TOTAL") != 1 or ten.count("Do not pad to ten:") != 1:
        raise ValueError("Expected the established ten-candidate equivalent expansion prompt")
    return ten.replace("up to TEN\nanswers TOTAL", "up to FIVE\nanswers TOTAL").replace(
        "Do not pad to ten:", "Do not pad to five:")
