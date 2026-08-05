"""
TOON (Token-Oriented Object Notation) Engine v2.2

Key Architectural Guarantees:
  1. Protocol Stability: Canonical form is standard for storage, docs, and LLM prompts.
  2. Non-Destructive Minification: `minify=True` compresses whitespace/formatting only; 
     it NEVER mangles key names or alters semantic data.
  3. API Symmetry: `loads()` / `decode_toon()` natively return Python objects (dict/list/scalar).
  4. Strict Validation: Detects list count mismatches, duplicate keys, and syntax errors.
"""

import json
import math
import re

__all__ = [
    "toon_encode",
    "toon_decode",
    "encode_toon",
    "decode_toon",
    "dumps",
    "loads",
    "ToonParseError",
]

_NUM_RE = re.compile(r'-?\d+(\.\d+)?([eE][+-]?\d+)?$')
_PRIMITIVE_LIKE = re.compile(r'(-?\d+(\.\d+)?([eE][+-]?\d+)?|true|false|null|nan|inf|-inf)$', re.IGNORECASE)
_SPECIALS = set('",:\n\t\r[]{} ')
_LIST_HEADER_RE = re.compile(r'^\[(\d+)\](\{(.*)\})?:(.*)$')
VALID_DUP_KEY_MODES = {"overwrite", "list", "error"}


class ToonParseError(ValueError):
    """Exception raised for syntax or constraint errors during TOON decoding."""
    def __init__(self, message, line_num=None, col_num=None, line_content=None):
        self.line_num = line_num
        self.col_num = col_num
        self.line_content = line_content
        loc_parts = []
        if line_num is not None:
            loc_parts.append(f"line {line_num}")
        if col_num is not None:
            loc_parts.append(f"col {col_num}")
        loc_str = f" at {', '.join(loc_parts)}" if loc_parts else ""
        full_msg = f"{message}{loc_str}"
        if line_content is not None:
            full_msg += f" -> {line_content!r}"
        super().__init__(full_msg)


# ---------------------------------------------------------------------------
# String & Primitive Utilities
# ---------------------------------------------------------------------------

def _needs_quoting(s):
    if s == "":
        return True
    if s != s.strip():
        return True
    if _PRIMITIVE_LIKE.fullmatch(s):
        return True
    if s in ("{}", "[]"):
        return True
    return any(c in _SPECIALS for c in s)


def _encode_str(s):
    if not _needs_quoting(s):
        return s
    return json.dumps(s, ensure_ascii=False)


def _format_float(obj, float_format=".17g"):
    if math.isnan(obj):
        return "nan"
    if math.isinf(obj):
        return "inf" if obj > 0 else "-inf"
    if float_format is None:
        return str(obj)
    elif callable(float_format):
        return float_format(obj)
    elif isinstance(float_format, str):
        formatted = format(obj, float_format)
        if "." not in formatted and "e" not in formatted.lower():
            formatted += ".0"
        return formatted
    return str(obj)


def _scan_quoted(content, i, strict=False, line_num=None):
    quote = content[i]
    j = i + 1
    n = len(content)
    while j < n:
        c = content[j]
        if c == '\\' and j + 1 < n:
            j += 2
            continue
        if c == quote:
            j += 1
            return content[i:j], j
        j += 1
    if strict:
        raise ToonParseError("Unterminated quoted string", line_num=line_num, col_num=i + 1, line_content=content)
    return content[i:n], n


def _unescape_quoted(raw_token, strict=False, line_num=None):
    if not raw_token:
        return ""
    quote = raw_token[0]
    inner = raw_token[1:-1] if raw_token[-1] == quote and len(raw_token) >= 2 else raw_token[1:]
    if quote == '"':
        try:
            return json.loads(raw_token)
        except (json.JSONDecodeError, ValueError):
            if strict:
                raise ToonParseError("Malformed escape sequence in string", line_num=line_num, line_content=raw_token)
    def _sub(m):
        c = m.group(1)
        return {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '"': '"', "'": "'"}.get(c, c)
    return re.sub(r'\\(.)', _sub, inner)


def _parse_val(token, strict=False, line_num=None):
    val = token.strip()
    if val == "" or val == "null":
        return None
    low = val.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if val == "{}":
        return {}
    if val == "[]":
        return []
    if low == "nan":
        return float('nan')
    if low == "inf" or low == "infinity":
        return float('inf')
    if low == "-inf" or low == "-infinity":
        return float('-inf')
    if len(val) >= 2 and val[0] in ('"', "'") and val[-1] == val[0]:
        return _unescape_quoted(val, strict=strict, line_num=line_num)
    if _NUM_RE.fullmatch(val):
        if "." in val or "e" in low:
            return float(val)
        return int(val)
    return val


def _split_fields(line, strict=False, line_num=None):
    """Split comma-separated fields while respecting quotes and backslash escaping."""
    out, buf, in_q, quote_ch, esc = [], [], False, '"', False
    for idx, ch in enumerate(line):
        if esc:
            buf.append(ch)
            esc = False
        elif in_q and ch == "\\":
            buf.append(ch)
            esc = True
        elif ch in ('"', "'") and (not in_q or ch == quote_ch):
            buf.append(ch)
            in_q = not in_q
            quote_ch = ch
        elif ch == "," and not in_q:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if in_q and strict:
        raise ToonParseError("Unterminated quote in comma-separated fields", line_num=line_num, line_content=line)
    out.append("".join(buf))
    return out


def _read_key(content, strict=False, line_num=None):
    if content[:1] in ('"', "'"):
        raw, end = _scan_quoted(content, 0, strict=strict, line_num=line_num)
        return _unescape_quoted(raw, strict=strict, line_num=line_num), content[end:].lstrip()
    j = 0
    n = len(content)
    while j < n and content[j] not in (':', '['):
        j += 1
    return content[:j].strip(), content[j:]


def _is_flat_dict(d):
    return isinstance(d, dict) and all(isinstance(v, (str, int, float, bool, type(None))) for v in d.values())


# ---------------------------------------------------------------------------
# Encoder Implementation
# ---------------------------------------------------------------------------

def _encode_scalar(obj, float_format=".17g"):
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, float):
        return _format_float(obj, float_format=float_format)
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, str):
        return _encode_str(obj)
    return _encode_str(str(obj))


def _uniform_cols(v):
    if not all(_is_flat_dict(x) for x in v):
        return None
    cols = list(v[0].keys())
    if all(list(x.keys()) == cols for x in v):
        return cols
    return None


def _encode_list_body(v, indent, indent_size, header_prefix, float_format=".17g", minify=False):
    prefix = (" " * indent_size) * indent
    cols = _uniform_cols(v)
    sep = "," if minify else ", "
    colon_space = ":" if minify else ": "

    if cols is not None:
        col_str = ",".join(_encode_str(str(c)) for c in cols)
        lines = [f"{prefix}{header_prefix}[{len(v)}]{{{col_str}}}:"]
        row_prefix = prefix + (" " * indent_size)
        for row in v:
            row_vals = [
                _encode_scalar(row.get(c), float_format=float_format) 
                if not isinstance(row.get(c), (dict, list)) 
                else _toon_encode_internal(row.get(c), 0, indent_size, float_format=float_format, minify=minify) 
                for c in cols
            ]
            lines.append(row_prefix + ",".join(row_vals))
        return lines

    if all(not isinstance(x, (dict, list)) for x in v):
        inline = sep.join(_toon_encode_internal(x, 0, indent_size, float_format=float_format, minify=minify) for x in v)
        return [f"{prefix}{header_prefix}[{len(v)}]{colon_space}{inline}"]
    
    lines = [f"{prefix}{header_prefix}[{len(v)}]:"]
    item_indent = indent + 1
    item_prefix = (" " * indent_size) * item_indent
    dash_space = "-" if minify else "- "
    for item in v:
        if isinstance(item, (dict, list)) and len(item) > 0:
            lines.append(f"{item_prefix}-")
            lines.append(_toon_encode_internal(item, item_indent + 1, indent_size, float_format=float_format, minify=minify))
        else:
            lines.append(f"{item_prefix}{dash_space}{_toon_encode_internal(item, 0, indent_size, float_format=float_format, minify=minify)}")
    return lines


def _toon_encode_internal(obj, indent, indent_size, float_format=".17g", minify=False):
    prefix = (" " * indent_size) * indent
    colon_space = ":" if minify else ": "
    
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        lines = []
        for k, v in obj.items():
            encoded_key = _encode_str(str(k))
            if isinstance(v, dict):
                if not v:
                    lines.append(f"{prefix}{encoded_key}:{colon_space if minify else ' '}}{{}}")
                else:
                    lines.append(f"{prefix}{encoded_key}:")
                    lines.append(_toon_encode_internal(v, indent + 1, indent_size, float_format=float_format, minify=minify))
            elif isinstance(v, (list, tuple)):
                if not v:
                    lines.append(f"{prefix}{encoded_key}:{colon_space if minify else ' '}}[]")
                else:
                    lines.extend(_encode_list_body(list(v), indent, indent_size, encoded_key, float_format=float_format, minify=minify))
            else:
                lines.append(f"{prefix}{encoded_key}{colon_space}{_toon_encode_internal(v, 0, indent_size, float_format=float_format, minify=minify)}")
        return "\n".join(lines)
    elif isinstance(obj, (list, tuple)):
        if not obj:
            return "[]"
        return "\n".join(_encode_list_body(list(obj), indent, indent_size, "", float_format=float_format, minify=minify))
    else:
        return _encode_scalar(obj, float_format=float_format)


def toon_encode(obj, minify=False, indent_size=2, float_format=".17g"):
    """
    Encodes a Python object into canonical TOON string.
    If `minify=True`, applies tight whitespace formatting without altering keys or data semantics.
    """
    effective_indent = 1 if minify else indent_size
    return _toon_encode_internal(obj, 0, effective_indent, float_format=float_format, minify=minify)


def encode_toon(json_data, minify=False, indent_size=2, float_format=".17g", **kwargs):
    obj = json.loads(json_data) if isinstance(json_data, str) else json_data
    return toon_encode(obj, minify=minify, indent_size=indent_size, float_format=float_format)


# ---------------------------------------------------------------------------
# Decoder Implementation
# ---------------------------------------------------------------------------

class _Line:
    __slots__ = ("indent", "content", "line_num")
    def __init__(self, indent, content, line_num):
        self.indent = indent
        self.content = content
        self.line_num = line_num


def _prep_lines(toon_str):
    text = toon_str.replace("\r\n", "\n").replace("\r", "\n")
    out = []
    for idx, raw in enumerate(text.split("\n"), start=1):
        if raw.strip() == "":
            continue
        stripped = raw.lstrip(" ")
        indent = len(raw) - len(stripped)
        out.append(_Line(indent, stripped.rstrip(), idx))
    return out


def _parse_row_fields(row_content, cols, strict=False, line_num=None):
    parts = _split_fields(row_content, strict=strict, line_num=line_num)
    row = {}
    for idx, col in enumerate(cols):
        row[col] = _parse_val(parts[idx], strict=strict, line_num=line_num) if idx < len(parts) else None
    return row


def _parse_cols(cols_str, strict=False, line_num=None):
    if cols_str == "":
        return []
    out = []
    for piece in _split_fields(cols_str, strict=strict, line_num=line_num):
        piece = piece.strip()
        if piece[:1] in ('"', "'"):
            out.append(_unescape_quoted(piece, strict=strict, line_num=line_num))
        else:
            out.append(piece)
    return out


def _assign_key(obj, key, value, on_duplicate_key="overwrite", line_num=None):
    if key in obj:
        if on_duplicate_key == "error":
            raise ToonParseError(f"Duplicate key detected: {key!r}", line_num=line_num)
        elif on_duplicate_key == "list":
            if not isinstance(obj[key], list):
                obj[key] = [obj[key]]
            obj[key].append(value)
        else:
            obj[key] = value
    else:
        obj[key] = value


def _parse_list_from_header(lines, i, header_match, base_indent, on_duplicate_key="overwrite", strict=False):
    header_line = lines[i]
    count = int(header_match.group(1))
    cols_str = header_match.group(3)
    tail = header_match.group(4).strip()

    if cols_str is not None:
        cols = _parse_cols(cols_str, strict=strict, line_num=header_line.line_num)
        result = []
        j = i + 1
        for _ in range(count):
            if j >= len(lines):
                break
            result.append(_parse_row_fields(lines[j].content, cols, strict=strict, line_num=lines[j].line_num))
            j += 1
        if strict and len(result) != count:
            raise ToonParseError(f"List count mismatch: declared [{count}], parsed {len(result)} rows", line_num=header_line.line_num)
        return result, j

    if tail != "":
        vals = [_parse_val(v, strict=strict, line_num=header_line.line_num) for v in _split_fields(tail, strict=strict, line_num=header_line.line_num)]
        if strict and len(vals) != count:
            raise ToonParseError(f"List count mismatch: declared [{count}], parsed {len(vals)} inline elements", line_num=header_line.line_num)
        return vals, i + 1

    result = []
    j = i + 1
    while j < len(lines):
        line = lines[j]
        if line.indent < base_indent:
            break
        if line.content == "-":
            if j + 1 < len(lines) and lines[j + 1].indent > line.indent:
                value, j = _parse_block(lines, j + 1, lines[j + 1].indent, on_duplicate_key=on_duplicate_key, strict=strict)
            else:
                value, j = {}, j + 1
            result.append(value)
        elif line.content.startswith("- ") or line.content.startswith("-"):
            val_str = line.content[2:] if line.content.startswith("- ") else line.content[1:]
            result.append(_parse_val(val_str, strict=strict, line_num=line.line_num))
            j += 1
        else:
            break
            
    if strict and len(result) != count:
        raise ToonParseError(f"List count mismatch: declared [{count}], parsed {len(result)} items", line_num=header_line.line_num)
        
    return result, j


def _parse_block(lines, i, base_indent, on_duplicate_key="overwrite", strict=False):
    if i >= len(lines) or lines[i].indent < base_indent:
        return {}, i

    if lines[i].content.startswith("-"):
        result = []
        j = i
        while j < len(lines) and lines[j].indent == base_indent and lines[j].content.startswith("-"):
            line = lines[j]
            if line.content == "-":
                if j + 1 < len(lines) and lines[j + 1].indent > line.indent:
                    value, j = _parse_block(lines, j + 1, lines[j + 1].indent, on_duplicate_key=on_duplicate_key, strict=strict)
                else:
                    value, j = {}, j + 1
                result.append(value)
            else:
                val_str = line.content[2:] if line.content.startswith("- ") else line.content[1:]
                result.append(_parse_val(val_str, strict=strict, line_num=line.line_num))
                j += 1
        return result, j

    m_top = _LIST_HEADER_RE.match(lines[i].content)
    if m_top:
        return _parse_list_from_header(lines, i, m_top, base_indent, on_duplicate_key=on_duplicate_key, strict=strict)

    obj = {}
    j = i
    while j < len(lines) and lines[j].indent == base_indent:
        line = lines[j]
        content = line.content
        key, rest = _read_key(content, strict=strict, line_num=line.line_num)
        
        if rest[:1] == "[":
            m = _LIST_HEADER_RE.match(rest)
            if not m:
                if strict:
                    raise ToonParseError("Malformed list header specification", line_num=line.line_num, line_content=content)
                j += 1
                continue
            value, j = _parse_list_from_header(lines, j, m, base_indent, on_duplicate_key=on_duplicate_key, strict=strict)
            _assign_key(obj, key, value, on_duplicate_key=on_duplicate_key, line_num=line.line_num)
            continue
            
        if rest[:1] == ":":
            val_part = rest[1:].strip()
            if val_part == "":
                if j + 1 < len(lines) and lines[j + 1].indent > base_indent:
                    value, j = _parse_block(lines, j + 1, lines[j + 1].indent, on_duplicate_key=on_duplicate_key, strict=strict)
                else:
                    value, j = {}, j + 1
                _assign_key(obj, key, value, on_duplicate_key=on_duplicate_key, line_num=line.line_num)
            else:
                _assign_key(obj, key, _parse_val(val_part, strict=strict, line_num=line.line_num), on_duplicate_key=on_duplicate_key, line_num=line.line_num)
                j += 1
            continue
            
        if strict:
            raise ToonParseError("Unrecognized key-value or list structure", line_num=line.line_num, line_content=content)
        j += 1
        
    return obj, j


def toon_decode(toon_str, on_duplicate_key="overwrite", strict=False):
    """
    Decodes a TOON formatted string into standard Python structures (dict, list, or scalar).
    """
    if on_duplicate_key not in VALID_DUP_KEY_MODES:
        raise ValueError(f"Invalid on_duplicate_key mode {on_duplicate_key!r}. Must be one of {sorted(VALID_DUP_KEY_MODES)}")
        
    lines = _prep_lines(toon_str)
    if not lines:
        return {}
        
    # Symmetric Top-Level Scalar Handling
    if len(lines) == 1:
        first = lines[0].content
        if first not in ("{}", "[]"):
            m = _LIST_HEADER_RE.match(first)
            if not m and ":" not in first and not first.startswith("-"):
                return _parse_val(first, strict=strict, line_num=lines[0].line_num)

    if len(lines) == 1 and lines[0].content in ("{}", "[]"):
        return {} if lines[0].content == "{}" else []

    first = lines[0].content
    m = _LIST_HEADER_RE.match(first)
    if m:
        root, _ = _parse_list_from_header(lines, 0, m, lines[0].indent, on_duplicate_key=on_duplicate_key, strict=strict)
    else:
        root, _ = _parse_block(lines, 0, lines[0].indent, on_duplicate_key=on_duplicate_key, strict=strict)
            
    return root


def decode_toon(toon_str, on_duplicate_key="overwrite", strict=False, **kwargs):
    """Alias for toon_decode to ensure API symmetry across loads/decode_toon."""
    return toon_decode(toon_str, on_duplicate_key=on_duplicate_key, strict=strict)


dumps = toon_encode
loads = toon_decode