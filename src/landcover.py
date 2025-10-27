"""
CORINE Land Cover classification utilities.
"""


def get_clc_level1(code):
    """
    Map CORINE Land Cover Level 3 code to Level 1 category.

    Parameters
    ----------
    code : int
        CORINE Land Cover Level 3 classification code

    Returns
    -------
    str
        Level 1 category name

    Notes
    -----
    CORINE Land Cover Level 1 categories:
    - 100-199: Artificial Surfaces
    - 200-299: Agricultural Areas
    - 300-399: Forest & Semi-Natural Areas
    - 400-499: Wetlands
    - 500-599: Water Bodies

    Examples
    --------
    >>> get_clc_level1(211)
    '2. Agricultural Areas'
    >>> get_clc_level1(312)
    '3. Forest & Semi-Natural Areas'
    """
    if 100 <= code < 200:
        return '1. Artificial Surfaces'
    if 200 <= code < 300:
        return '2. Agricultural Areas'
    if 300 <= code < 400:
        return '3. Forest & Semi-Natural Areas'
    if 400 <= code < 500:
        return '4. Wetlands'
    if 500 <= code < 600:
        return '5. Water Bodies'
    return 'Other'
