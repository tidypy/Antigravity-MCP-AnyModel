import sys
import json
import math

# Import local TOON_V2 module directly
import TOON_V2 as m

passed, failed = [], []

def check(label, cond, extra=""):
    if cond:
        passed.append(label)
    else:
        failed.append((label, extra))
        print(f"FAIL: {label}  {extra}")

def rt(label, obj, **kw):
    enc = m.toon_encode(obj, **kw)
    dec = m.toon_decode(enc)
    
    # Primitive scalar equality check
    ok = (dec == obj)
    
    # Special-case NaN equality
    if not ok:
        try:
            if isinstance(obj, float) and math.isnan(obj) and isinstance(dec, float) and math.isnan(dec):
                ok = True
            elif isinstance(obj, dict) and isinstance(dec, dict) and obj.keys() == dec.keys():
                ok = all(
                    (isinstance(v, float) and math.isnan(v) and isinstance(dec[k], float) and math.isnan(dec[k]))
                    or v == dec[k]
                    for k, v in obj.items()
                )
        except Exception:
            pass
            
    check(label, ok, f"\n  obj={obj!r}\n  enc={enc!r}\n  dec={dec!r}")
    return enc, dec

# ---- Top-Level Primitives (Symmetry Verification) ----
rt("top_level_scalar_int", 42)
rt("top_level_scalar_float", 3.14159)
rt("top_level_scalar_str", "hello world")
rt("top_level_scalar_bool", True)
rt("top_level_scalar_none", None)

# ---- Complex Bug Cases (Expected to PASS) ----
rt("heterogeneous_dict_array", {"items": [{"id": 1, "name": "Alice"}, {"id": 2, "email": "bob@x.com"}]})
rt("nonflat_dict_array", {"items": [{"id": 1, "meta": {"a": 1}}, {"id": 2, "meta": {"a": 2}}]})
rt("mixed_list", {"items": [{"id": 1}, "just a string", 42]})
rt("list_multikey_dict", {"items": [{"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3}]})
rt("nested_lists", {"matrix": [[1, 2], [3, 4]]})
rt("empty_dict_value", {"a": {}, "b": 1})
rt("empty_list_value", {"a": [], "b": 1})
rt("empty_dict_in_row", {"items": [{"id": 1, "meta": {}}, {"id": 2, "meta": {}}]})
rt("special_keys", {"first name": "Alice", "user-id": 1, "a.b.c": True})
rt("indent_size_1", {"a": {"b": {"c": 1}}}, indent_size=1)
rt("indent_size_4", {"a": {"b": {"c": 1, "d": [1, 2, 3]}}}, indent_size=4)

enc, dec = rt("top_level_list_of_dicts", [{"id": 1}, {"id": 2}])
check("top_level_list_no_fake_items_key", "items" not in enc, enc)

rt("top_level_empty_list", [])
rt("top_level_empty_dict", {})
rt("deeply_nested", {"a": {"b": {"c": {"d": {"e": 1}}}}})
rt("list_of_lists_of_dicts", {"x": [[{"a": 1}], [{"b": 2}]]})
rt("dict_with_list_and_dict_siblings", {"a": [1, 2, 3], "b": {"x": 1}, "c": "plain"})

# Floats including special values
rt("float_nan", {"x": float('nan')})
rt("float_inf", {"x": float('inf')})
rt("float_neg_inf", {"x": float('-inf')})
rt("float_normal", {"x": 0.1, "y": 1e20, "z": 1e-10})

# Escape correctness (forced quoting via comma)
original = "a," + chr(92) + "n" + "b"
enc = m.toon_encode({"x": original})
dec = m.toon_decode(enc)
check("escape_literal_backslash_n", dec["x"] == original, f"enc={enc!r} dec={dec!r}")

# Control chars surviving splitlines()
weird = "line1" + chr(0x0c) + "line2"
enc = m.toon_encode({"x": weird})
dec = m.toon_decode(enc)
check("formfeed_survives", dec.get("x") == weird, f"enc={enc!r} dec={dec!r}")

weird2 = "a" + chr(0x0b) + "b"
enc = m.toon_encode({"x": weird2})
dec = m.toon_decode(enc)
check("vtab_survives", dec.get("x") == weird2, f"enc={enc!r} dec={dec!r}")

# Numeric-looking string should NOT become an int
enc = m.toon_encode({"id": "123_456"})
dec = m.toon_decode(enc)
check("underscore_numeric_string_stays_string", dec.get("id") == "123_456", f"enc={enc!r} dec={dec!r}")

# Key_map roundtrip (Fixed: dec is already a dict)
enc = m.toon_encode({"user_id": 1, "user_name": "Bob"}, key_map={"user_id": "uid", "user_name": "un"})
dec_obj = m.decode_toon(enc, key_map={"user_id": "uid", "user_name": "un"})
check("key_map_roundtrip", dec_obj == {"user_id": 1, "user_name": "Bob"}, f"enc={enc!r} dec={dec_obj!r}")

# Strip_nulls regression check
enc = m.toon_encode({"a": 1, "b": None}, strip_nulls=True)
check("strip_nulls", "b" not in enc, enc)

# Dumps/loads aliases exist and work natively
check("dumps_alias_exists", hasattr(m, "dumps"))
check("loads_alias_exists", hasattr(m, "loads"))
check("dumps_loads_roundtrip", m.loads(m.dumps({"a": 1})) == {"a": 1})

# Backward-compatibility: Tabular output assertions
sample = {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
enc = m.toon_encode(sample)
check("bc_tabular_header", "users[2]{id,name}:" in enc, enc)
check("bc_tabular_row", "1,Alice" in enc, enc)
dec = m.toon_decode("users[2]{id,name}:\n1,Alice\n2,Bob")
check("bc_tabular_decode", dec == sample, dec)

enc2 = m.toon_encode({"user_id": 101, "user_name": "Alice", "extra": None}, strip_nulls=True, key_map={"user_id": "uid", "user_name": "un"})
check("bc_sweet_spot_uid", "uid: 101" in enc2, enc2)
check("bc_sweet_spot_un", "un: Alice" in enc2, enc2)
check("bc_sweet_spot_no_extra", "extra" not in enc2, enc2)

print(f"\n{'='*60}\nPASSED: {len(passed)}   FAILED: {len(failed)}")
if failed:
    print("\nFAILED CASES:")
    for label, extra in failed:
        print(f" - {label}")
    sys.exit(1)
else:
    print("ALL PASS")