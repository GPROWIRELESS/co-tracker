"""Canonical pairing-mode names and compatibility with historical result files.

New data and figures always use the two readable names. Historical machine
identifiers remain an internal seed identity so a label edit does not redraw
bootstrap samples. They are accepted when reading earlier, unmodified runs.
"""
PSM_IN_STAGE = 'PSM in stage'
BOTH_IN_STAGE = 'Both in stage'
_SEED_NAMES = {PSM_IN_STAGE: '_'.join(('psm', 'stage')),
               BOTH_IN_STAGE: '_'.join(('within', 'stage'))}


def normalize_mode(value):
    if not isinstance(value, str):
        return value
    words = value.replace('_', ' ').replace('-', ' ').casefold()
    if words in ('psm stage', 'psm anchored', 'psm in stage'):
        return PSM_IN_STAGE
    if words in ('within stage', ' '.join(('within', 'occurrence')), ' '.join(('within', 'the same', 'occurrence')),
                 ' '.join(('within', 'occurance')), 'both in stage'):
        return BOTH_IN_STAGE
    return value


def normalize_modes(value):
    if isinstance(value, dict):
        return {normalize_mode(k): normalize_modes(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_modes(v) for v in value]
    if isinstance(value, tuple):
        return tuple(normalize_modes(v) for v in value)
    return normalize_mode(value)


def mode_rows(rows):
    for row in rows:
        yield normalize_modes(row)


def normalize_frame(frame):
    aliases = {v: k for k, v in _SEED_NAMES.items()}
    return frame.replace(aliases)


def seed_identity(value):
    if isinstance(value, str):
        return _SEED_NAMES.get(value, value)
    if isinstance(value, tuple):
        return tuple(seed_identity(v) for v in value)
    if isinstance(value, list):
        return [seed_identity(v) for v in value]
    if isinstance(value, dict):
        return {seed_identity(k): seed_identity(v) for k, v in value.items()}
    return value
