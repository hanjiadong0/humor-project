"""Check for pattern mismatches between JOKE_PATTERNS and PATTERN_DEFINITIONS"""

from patterns import JOKE_PATTERNS, PATTERN_DEFINITIONS

print("=" * 80)
print("PATTERN VALIDATION REPORT")
print("=" * 80)

print(f"\nTotal patterns in JOKE_PATTERNS list: {len(JOKE_PATTERNS)}")
print(f"Total patterns in PATTERN_DEFINITIONS dict: {len(PATTERN_DEFINITIONS)}")

# Find patterns in list but not in dict
missing_in_dict = [p for p in JOKE_PATTERNS if p not in PATTERN_DEFINITIONS]

print("\n" + "=" * 80)
if missing_in_dict:
    print(f"ISSUE: {len(missing_in_dict)} patterns in JOKE_PATTERNS are NOT in PATTERN_DEFINITIONS:")
    for i, pattern in enumerate(missing_in_dict, 1):
        print(f"  {i}. {pattern}")
else:
    print("GOOD: All patterns in JOKE_PATTERNS exist in PATTERN_DEFINITIONS")

# Find patterns in dict but not in list
extra_in_dict = [p for p in PATTERN_DEFINITIONS if p not in JOKE_PATTERNS]

print("\n" + "=" * 80)
if extra_in_dict:
    print(f"NOTE: {len(extra_in_dict)} patterns in PATTERN_DEFINITIONS are NOT in JOKE_PATTERNS:")
    for i, pattern in enumerate(extra_in_dict, 1):
        print(f"  {i}. {pattern}")
else:
    print("GOOD: All patterns in PATTERN_DEFINITIONS are in JOKE_PATTERNS")

print("\n" + "=" * 80)
print("ALL PATTERNS IN JOKE_PATTERNS:")
print("=" * 80)
for i, pattern in enumerate(JOKE_PATTERNS, 1):
    status = "OK" if pattern in PATTERN_DEFINITIONS else "MISSING"
    print(f"{i:2d}. [{status}] {pattern}")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
if missing_in_dict:
    print("Fix the pattern names in JOKE_PATTERNS to match PATTERN_DEFINITIONS")
    print("Or add missing patterns to PATTERN_DEFINITIONS")
else:
    print("Pattern names are consistent - no action needed")
