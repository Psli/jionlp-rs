//! Chinese money parser — simplified port of `jionlp/gadget/money_parser.py`.
//!
//! Handles:
//!   "100元", "1,000.50元"         — Arabic numerals with unit
//!   "一百元", "三千五百万元"       — Chinese numerals (reuses char2num)
//!   "1.5万元", "一点五万美元"      — mixed decimal + big unit
//!   "100美元", "100 USD"           — alternative currency cases
//!   "人民币100元", "RMB 100"       — currency prefix
//!   "￥100", "$100.5"              — currency symbols (since round 6)
//!   "100元5角3分"                  — 元/角/分 triple (since round 6)
//!   "约100元", "100元左右"          — blur modifier (since round 6)
//!   "100-200元", "100到200元"      — numeric range (since round 6)
//!
//! Not yet supported (tracked in PLAN.md):
//!   * Multiple currencies in one input
//!   * 合计/共 / 负数的精细表达
//!
//! Output is a [`MoneyInfo`] with a normalized numeric value, the currency
//! case name (元 / 美元 / ...), a `definition` ("accurate" / "blur") and
//! optionally an `end_num` for ranges.

use crate::gadget::money_num2char::char2num;
use once_cell::sync::Lazy;
use regex::Regex;

#[derive(Debug, Clone, PartialEq)]
pub struct MoneyInfo {
    /// Normalized to the base currency unit (元, 美元 etc.) as `f64`. For
    /// ranges this is the lower bound.
    pub num: f64,
    /// Currency name, e.g. "元", "美元", "港币".
    pub case: String,
    /// "accurate" or "blur" (set by modifiers 约/大约/左右).
    pub definition: &'static str,
    /// Upper bound for ranges ("100-200元"); `None` for single values.
    pub end_num: Option<f64>,
}

const DEFAULT_UNIT: &str = "元";

/// Currency symbols that can appear as a prefix or suffix.
const CURRENCY_SYMBOLS: &[(&str, &str)] = &[
    ("￥", "元"),
    ("¥", "元"),
    ("$", "美元"),
    ("USD$", "美元"),
    ("€", "欧元"),
    ("£", "英镑"),
    ("HK$", "港币"),
];

const BLUR_PREFIXES: &[&str] = &["大约", "大概", "约", "约莫", "约摸", "差不多"];
const BLUR_SUFFIXES: &[&str] = &["左右", "上下", "许", "许多"];
#[allow(dead_code)]
const RANGE_SEPS: &[&str] = &["到", "至", "-", "—", "--", "~", "～", "–"];
/// Lower-bound modifiers — the parsed value is the floor.
const BLUR_LO_PREFIXES: &[&str] = &["至少", "不少于", "超过", "起码", "大于", "多于", "最少"];
const BLUR_LO_SUFFIXES: &[&str] = &["以上", "(含)以上", "（含）以上", "多"];
/// Upper-bound modifiers — the parsed value is the ceiling.
const BLUR_HI_PREFIXES: &[&str] = &["近", "接近", "将近", "不到", "不足", "少于", "小于", "最多"];
const BLUR_HI_SUFFIXES: &[&str] = &["以下", "(含)以下", "（含）以下"];

// Currency-name table. Longest first so the greedy match picks "人民币" over
// a potential "人" alone (not that "人" is a currency, but the ordering
// discipline matters for more ambiguous cases like "新加坡元" vs "元").
const CURRENCY_CASES: &[(&str, &str)] = &[
    ("新加坡元", "新加坡元"),
    ("新台币", "新台币"),
    ("人民币", "元"),
    ("港元", "港元"),
    ("港币", "港元"), // 港币 → canonical 港元 (Python convention)
    ("台币", "新台币"),
    ("泰铢", "泰铢"),
    ("美元", "美元"),
    ("欧元", "欧元"),
    ("日元", "日元"),
    ("日币", "日元"),
    ("英镑", "英镑"),
    ("韩元", "韩元"),
    ("卢布", "卢布"),
    ("美金", "美元"),
    ("澳元", "澳元"),
    ("加元", "加元"),
    ("RMB", "元"),
    ("USD", "美元"),
    ("EUR", "欧元"),
    ("JPY", "日元"),
    ("GBP", "英镑"),
    ("块钱", "元"),
    ("圆整", "元"), // 大写金额 trailing — 肆佰叁拾萬圆整
    ("圆", "元"),   // 肆佰叁拾萬圆
    ("元", "元"),
    ("块", "元"),
    ("毛", "元"), // 十块三毛 — 毛 itself is a fractional word, kept as 元
];

// Unit multipliers (only the integer-scale suffixes; 分/角 and 元 are handled
// separately).
const UNIT_MULTIPLIERS: &[(&str, f64)] = &[
    ("兆", 1e12),
    ("亿", 1e8),
    ("万", 1e4),
    ("萬", 1e4),
    ("千", 1e3),
    ("百", 1e2),
];

static PURE_NUMBER: Lazy<Regex> = Lazy::new(|| Regex::new(r"^-?\d+(\.\d+)?$").unwrap());
static LEADING_NUMBER: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^-?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?").unwrap());

/// Parse a money string. Returns `None` if it can't be interpreted.
pub fn parse_money(s: &str) -> Option<MoneyInfo> {
    parse_money_with_default(s, DEFAULT_UNIT)
}

/// Same but lets the caller pick the default currency when the input has no
/// case annotation (e.g. passing in "美元" makes bare "100" parse as USD).
pub fn parse_money_with_default(s: &str, default_unit: &str) -> Option<MoneyInfo> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return None;
    }

    // Input normalization — run each of these BEFORE any parser path:
    //   `，` → `,`                        fullwidth-comma as thousands sep
    //   strip `（含）` / `(含)` tokens     Python ignores these annotations
    //   trim leading `从`/`自`             Python's 起始介词 prefix
    //   collapse internal whitespace       `新台币 177.1 亿元` → `新台币177.1亿元`
    //   `——` / `--` kept as separators (handled in try_parse_range).
    let normalized: String = trimmed
        .replace('，', ",")
        .replace("（含）", "")
        .replace("(含)", "");
    let normalized = normalized.trim();
    let normalized = normalized
        .strip_prefix('从')
        .or_else(|| normalized.strip_prefix('自'))
        .unwrap_or(normalized);
    // Collapse inner whitespace so `新台币 177.1 亿元` matches.
    let collapsed: String = normalized
        .split_ascii_whitespace()
        .collect::<Vec<_>>()
        .join("");
    let collapsed = collapsed.replace(['\u{3000}', ' '], "");
    // `个` as numeric filler between digit and big-unit (`两个亿` → `两亿`,
    // `三个万` → `三万`) is treated as optional by Python.
    let collapsed = collapsed
        .replace("个亿", "亿")
        .replace("个万", "万")
        .replace("个千", "千")
        .replace("个百", "百")
        // Money-specific CN variants: 仟→千, 萬→万, 佰→百, 拾→十.
        .replace('仟', "千")
        .replace('萬', "万")
        .replace('佰', "百")
        .replace('拾', "十");
    let trimmed = collapsed.as_str();

    // 1) Detect blur modifiers (约/大约/左右/近/至少/…) and strip them.
    //    Returns one of: None, "blur", "blur+" (lower-bound), "blur-" (upper-bound).
    let (modifier, core) = strip_blur_modifiers(trimmed);

    // 2) Try as a range first.
    if let Some(info) = try_parse_range(core, default_unit) {
        let def = match modifier {
            Some(_) => "blur",
            None => "blur", // ranges are always blur
        };
        return Some(MoneyInfo {
            definition: def,
            ..info
        });
    }

    // Try blur-quantifier shapes first: `十几块` / `几十万块` / `八九亿韩元`.
    if let Some(r) = try_blur_quantifier(core, default_unit) {
        return Some(r);
    }

    let definition = modifier.unwrap_or("accurate");
    let mut result = parse_single_money(core, default_unit, definition)?;
    // `多` marker inside a numeric body upgrades to `blur` AND extends the
    // value into a range `[N, next_higher_digit_round]`. E.g.
    //   `3000多` → Range(3000, 4000);  `70多` → Range(70, 80);
    //   `500多` → Range(500, 600);  `十多` → Range(10, 20) (handled by
    //   quantifier path separately).
    if definition == "accurate" && trimmed.contains('多') {
        result.definition = "blur";
        if result.end_num.is_none() {
            let n = result.num;
            let hi = next_duo_ceiling(n);
            if hi > n {
                result.end_num = Some(hi);
            }
        }
    }
    Some(result)
}

/// Map `N` to the next-digit ceiling for `多`-blur expansion.
/// Examples:
///   3000 → 4000,  3500 → 4000,  70 → 80,  100 → 200,
///   10 → 20,      50 → 60,      1000 → 2000.
fn next_duo_ceiling(n: f64) -> f64 {
    if n <= 0.0 {
        return n;
    }
    // Round up the leading digit by 1. E.g. 3000 → 4000.
    let magnitude = 10f64.powi((n.log10().floor()) as i32);
    let leading = (n / magnitude).floor();
    (leading + 1.0) * magnitude
}

/// Detect modifier: returns (Some("blur"|"blur+"|"blur-"), stripped_core).
/// Iterates until no more modifier sticks, so compound forms like
/// `至少X以上` (two bounds-markers) and `约X多` (blur + blur) work.
fn strip_blur_modifiers(s: &str) -> (Option<&'static str>, &str) {
    let mut cur = s;
    let mut tag: Option<&'static str> = None;
    loop {
        let before = cur;
        // Strong lower-bound (prefix then suffix).
        for p in BLUR_LO_PREFIXES {
            if let Some(rest) = cur.strip_prefix(*p) {
                tag = Some("blur+");
                cur = rest.trim_start();
                break;
            }
        }
        for p in BLUR_HI_PREFIXES {
            if let Some(rest) = cur.strip_prefix(*p) {
                tag = Some("blur-");
                cur = rest.trim_start();
                break;
            }
        }
        for sfx in BLUR_LO_SUFFIXES {
            if let Some(rest) = cur.strip_suffix(*sfx) {
                tag = Some("blur+");
                cur = rest.trim_end();
                break;
            }
        }
        for sfx in BLUR_HI_SUFFIXES {
            if let Some(rest) = cur.strip_suffix(*sfx) {
                tag = Some("blur-");
                cur = rest.trim_end();
                break;
            }
        }
        for p in BLUR_PREFIXES {
            if let Some(rest) = cur.strip_prefix(*p) {
                if tag.is_none() {
                    tag = Some("blur");
                }
                cur = rest.trim_start();
                break;
            }
        }
        for sfx in BLUR_SUFFIXES {
            if let Some(rest) = cur.strip_suffix(*sfx) {
                if tag.is_none() {
                    tag = Some("blur");
                }
                cur = rest.trim_end();
                break;
            }
        }
        if cur == before {
            break;
        }
    }
    (tag, cur)
}

fn parse_single_money(s: &str, default_unit: &str, definition: &'static str) -> Option<MoneyInfo> {
    let s = s.trim();

    // Try currency symbol (prefix/suffix) first — one char replaces a case name.
    let (sym_case, remaining) = strip_symbol(s);

    // Strip leading case name ("人民币"/"港币").
    let (prefix_case, after_prefix) = strip_leading_case(remaining);

    // 元角分 compound (e.g. "100元5角3分") — treat specially, only in pure-yuan context.
    if let Some(num) = parse_yuan_jiao_fen(after_prefix) {
        return Some(MoneyInfo {
            num,
            case: sym_case
                .or(prefix_case)
                .unwrap_or_else(|| default_unit.to_string()),
            definition,
            end_num: None,
        });
    }

    // Strip trailing case name ("元" / "美元") from remainder.
    let (suffix_case, body) = strip_trailing_case(after_prefix);
    // After the first suffix strip, the body may still carry an embedded
    // 元/块 marker from the money amount itself (`三万元欧元` → suffix=欧元,
    // body=三万元; `9000元日币` → suffix=日元, body=9000元). Peel it once more.
    let (body, had_yuan_inner) = match strip_trailing_case(body) {
        (Some(inner), rest) if inner == "元" => (rest, true),
        _ => (body, false),
    };
    let _ = had_yuan_inner;

    // Priority: prefix_case beats suffix_case when both exist AND both map to
    // compatible currency roots (`港币两千九百六十元` → prefer 港元, not 元).
    // Otherwise pick the more specific one.
    let case = match (prefix_case.as_deref(), suffix_case.as_deref()) {
        (Some(p), Some(s)) if s == "元" => p.to_string(),
        _ => suffix_case
            .clone()
            .or(prefix_case.clone())
            .or(sym_case.clone())
            .unwrap_or_else(|| default_unit.to_string()),
    };

    let num = parse_number_body(body)?;
    // Python rounds to 2 decimals (currency precision).
    let num = (num * 100.0).round() / 100.0;
    Some(MoneyInfo {
        num,
        case,
        definition,
        end_num: None,
    })
}

/// Detect and strip blur modifiers. Returns `(is_blur, trimmed_core)`.
/// Kept for tests; `strip_blur_modifiers` is the primary entry in the parser.
#[allow(dead_code)]
fn strip_blur(s: &str) -> (bool, &str) {
    for p in BLUR_PREFIXES {
        if let Some(rest) = s.strip_prefix(p) {
            return (true, rest.trim_start());
        }
    }
    for sfx in BLUR_SUFFIXES {
        if let Some(rest) = s.strip_suffix(sfx) {
            return (true, rest.trim_end());
        }
    }
    (false, s)
}

/// Strip a currency symbol from either end of `s`. Multi-char symbols (HK$)
/// are tried before single-char ($).
fn strip_symbol(s: &str) -> (Option<String>, &str) {
    for (sym, canonical) in CURRENCY_SYMBOLS {
        if let Some(rest) = s.strip_prefix(*sym) {
            return (Some(canonical.to_string()), rest.trim_start());
        }
        if let Some(rest) = s.strip_suffix(*sym) {
            return (Some(canonical.to_string()), rest.trim_end());
        }
    }
    (None, s)
}

fn try_parse_range(s: &str, default_unit: &str) -> Option<MoneyInfo> {
    // Longest separator first so `——`/`--` beat `—`/`-`.
    const SEP_ORDER: &[&str] = &["——", "--", "到", "至", "—", "-", "~", "～", "–"];

    for sep in SEP_ORDER {
        if let Some((left_raw, right_raw)) = split_once_meaningful(s, sep) {
            if !has_any_digit(left_raw) || !has_any_digit(right_raw) {
                continue;
            }
            // Per-side: strip any embedded currency/symbol/leading case so
            // forms like `1万元--5万元` or `从8500到3万港元` both work. Then
            // peel an inner trailing `元` (money marker) if the first strip
            // already consumed the actual currency name (`两千万元人民币`
            // → suffix 人民币 → body 两千万元 → peel 元 → 两千万).
            let (l_sym, l1) = strip_symbol(left_raw.trim());
            let (l_pre, l2) = strip_leading_case(l1);
            let (l_suf, l_body0) = strip_trailing_case(l2);
            let l_body = match strip_trailing_case(l_body0) {
                (Some(inner), rest) if inner == "元" => rest,
                _ => l_body0,
            };
            let (r_sym, r1) = strip_symbol(right_raw.trim());
            let (r_pre, r2) = strip_leading_case(r1);
            let (r_suf, r_body0) = strip_trailing_case(r2);
            let r_body = match strip_trailing_case(r_body0) {
                (Some(inner), rest) if inner == "元" => rest,
                _ => r_body0,
            };

            let case = r_suf
                .or(l_suf)
                .or(r_pre)
                .or(l_pre)
                .or(r_sym)
                .or(l_sym)
                .unwrap_or_else(|| default_unit.to_string());

            // Parse each side independently.
            let mut lo = parse_number_body(l_body.trim())?;
            let hi = parse_number_body(r_body.trim())?;

            // Right-side unit propagation. If right ends in a big-unit that
            // left doesn't *contain*, apply that multiplier to left.
            //   `两到三万`     → l=两 lacks 万 → lo *= 万 = 2e4.
            //   `1万-5万`     → l=1万 has 万 → no propagation.
            //   `七千到九千亿` → l=七千 lacks 亿 (inner 千 doesn't count) → lo *= 亿 = 7e11.
            //   `8500到3万`   → l=Arabic → Python keeps 8500 intact (guard).
            // Check for a k/w casual-suffix on right side too.
            let r_casual_mult = if r_body.ends_with('k') || r_body.ends_with('K') {
                Some(("k", 1e3))
            } else if r_body.ends_with('w') || r_body.ends_with('W') {
                Some(("w", 1e4))
            } else {
                None
            };
            let r_outer: Option<(&str, f64)> = UNIT_MULTIPLIERS
                .iter()
                .find(|(u, _)| r_body.ends_with(*u))
                .map(|(u, m)| (*u, *m))
                .or(r_casual_mult);
            let l_has_arabic = l_body.chars().any(|c| c.is_ascii_digit());
            if let Some((r_char, r_mul)) = r_outer {
                let l_already = l_body.contains(r_char);
                // For Arabic left, only propagate when the ratio hi/lo is
                // at least r_mul — indicating the left value is magnitudes
                // smaller than right and thus missing the unit. Blocks
                // `8500到3万港元` (ratio 3.5) while allowing `2——3万港币`
                // (ratio 15000).
                let ratio_implies_missing = l_has_arabic && lo > 0.0 && (hi / lo) >= r_mul;
                let cn_left_needs_unit = !l_has_arabic && !l_already;
                if cn_left_needs_unit || ratio_implies_missing {
                    lo *= r_mul;
                }
            }

            return Some(MoneyInfo {
                num: lo,
                case,
                definition: "accurate",
                end_num: Some(hi),
            });
        }
    }
    None
}

fn split_once_meaningful<'a>(s: &'a str, sep: &str) -> Option<(&'a str, &'a str)> {
    // Skip cases where the separator is at the start (e.g. "-100" = negative).
    let idx = s.find(sep)?;
    if idx == 0 {
        return None;
    }
    let left = &s[..idx];
    let right = &s[idx + sep.len()..];
    if right.is_empty() {
        return None;
    }
    Some((left, right))
}

fn has_any_digit(s: &str) -> bool {
    s.chars().any(|c| {
        c.is_ascii_digit()
            || matches!(
                c,
                '零' | '一'
                    | '二'
                    | '两'
                    | '三'
                    | '四'
                    | '五'
                    | '六'
                    | '七'
                    | '八'
                    | '九'
                    | '壹'
                    | '贰'
                    | '叁'
                    | '肆'
                    | '伍'
                    | '陆'
                    | '柒'
                    | '捌'
                    | '玖'
                    | '十'
                    | '拾'
                    | '百'
                    | '佰'
                    | '千'
                    | '仟'
                    | '万'
                    | '萬'
                    | '亿'
                    | '兆'
            )
    })
}

/// Try to interpret `s` as `<Y>元<J>角<F>分` — returns the decimal-yuan value.
/// Accepts any subset, e.g. "100元5角", "100元", "3分" (3分 → 0.03元).
///
/// Semantics: each suffix consumes the digit run that *immediately precedes*
/// it (Python style). So in "100元5角3分":
///   fen  ← "3"  (digits before 分)
///   jiao ← "5"  (digits between 元 and 角)
///   yuan ← "100" (digits before 元)
fn parse_yuan_jiao_fen(s: &str) -> Option<f64> {
    let has_yuan = s.contains("元") || s.contains("块");
    let has_jiao = s.contains("角") || s.contains("毛");
    let has_fen = s.contains("分");
    // Need at least two of the three present to count as compound — a bare
    // "100元" should be handled by the Arabic-number path, not this one.
    let present = (has_yuan as u8) + (has_jiao as u8) + (has_fen as u8);
    if present < 2 {
        return None;
    }

    let mut remainder = s;
    let mut fen = 0.0;
    let mut jiao = 0.0;
    let mut yuan = 0.0;

    // Consume from the right: 分, then 角/毛, then 元/块.
    if let Some((left, _)) = rsplit_once_any(remainder, &["分"]) {
        if !left.is_empty() {
            // Now scan left for the digit-run boundary (a marker char like 元/角/块/毛
            // stops the run).
            let (prefix, digits) = carve_trailing_number(left);
            if !digits.is_empty() {
                fen = parse_number_body(digits)?;
            }
            remainder = prefix;
        }
    }
    if let Some((left, _)) = rsplit_once_any(remainder, &["角", "毛"]) {
        if !left.is_empty() {
            let (prefix, digits) = carve_trailing_number(left);
            if !digits.is_empty() {
                jiao = parse_number_body(digits)?;
            }
            remainder = prefix;
        }
    }
    if let Some((left, _)) = rsplit_once_any(remainder, &["元", "块"]) {
        if !left.is_empty() {
            yuan = parse_number_body(left.trim())?;
        }
        // Trailing text after 元 was consumed above; discard anything left.
    }

    Some(yuan + jiao * 0.1 + fen * 0.01)
}

/// Split at the LAST occurrence of any separator in `seps`.
fn rsplit_once_any<'a>(s: &'a str, seps: &[&str]) -> Option<(&'a str, &'a str)> {
    let mut best: Option<(usize, usize)> = None; // (start, sep_len)
    for sep in seps {
        if let Some(idx) = s.rfind(sep) {
            match best {
                Some((b, _)) if idx <= b => {}
                _ => best = Some((idx, sep.len())),
            }
        }
    }
    let (idx, len) = best?;
    Some((&s[..idx], &s[idx + len..]))
}

/// Walk backwards through `s` and return (prefix, digits) where `digits`
/// is the trailing contiguous run of numeric characters (ASCII digits plus
/// Chinese numerals like 一二三 and 零点十). Whitespace is treated as
/// non-numeric and terminates the run.
fn carve_trailing_number(s: &str) -> (&str, &str) {
    let mut split_at = s.len();
    for (idx, ch) in s.char_indices().rev() {
        let is_num = ch.is_ascii_digit()
            || ch == '.'
            || matches!(
                ch,
                '零' | '〇'
                    | '一'
                    | '二'
                    | '三'
                    | '四'
                    | '五'
                    | '六'
                    | '七'
                    | '八'
                    | '九'
                    | '壹'
                    | '贰'
                    | '叁'
                    | '肆'
                    | '伍'
                    | '陆'
                    | '柒'
                    | '捌'
                    | '玖'
                    | '两'
                    | '十'
                    | '百'
                    | '千'
                    | '万'
                    | '亿'
                    | '拾'
                    | '佰'
                    | '仟'
            );
        if is_num {
            split_at = idx;
        } else {
            break;
        }
    }
    (&s[..split_at], &s[split_at..])
}

fn strip_leading_case(text: &str) -> (Option<String>, &str) {
    for (pat, canonical) in CURRENCY_CASES {
        if let Some(rest) = text.strip_prefix(pat) {
            let rest = rest.trim_start();
            return (Some(canonical.to_string()), rest);
        }
    }
    (None, text)
}

fn strip_trailing_case(text: &str) -> (Option<String>, &str) {
    for (pat, canonical) in CURRENCY_CASES {
        if let Some(stripped) = text.strip_suffix(pat) {
            let stripped = stripped.trim_end();
            return (Some(canonical.to_string()), stripped);
        }
    }
    (None, text)
}

fn parse_number_body(body: &str) -> Option<f64> {
    let body = body.trim();
    if body.is_empty() {
        return None;
    }
    // Tolerate trailing blur marker "多" (e.g. `3000多欧元` → 3000).
    let body = body.strip_suffix('多').unwrap_or(body).trim();

    // `k` / `w` / `千` / `万` single-char suffix shorthand used in casual
    // Chinese business writing (e.g. `15k`, `30w`).
    if let Some(rest) = body.strip_suffix(|c: char| c == 'k' || c == 'K') {
        if let Some(v) = parse_number_body(rest) {
            return Some(v * 1e3);
        }
    }
    if let Some(rest) = body.strip_suffix(|c: char| c == 'w' || c == 'W') {
        if let Some(v) = parse_number_body(rest) {
            return Some(v * 1e4);
        }
    }
    // `仟` is a CN big-money variant of 千.
    let body = &body.replace('仟', "千");
    let body = body.as_str();

    // Case A: leading Arabic numeric, possibly followed by big-unit suffix.
    if let Some(m) = LEADING_NUMBER.find(body) {
        let leading = &body[..m.end()];
        let tail = body[m.end()..].trim();
        let value: f64 = leading.replace(',', "").parse().ok()?;
        if tail.is_empty() {
            return Some(value);
        }
        // Big-unit multiplier on Arabic number: e.g. "1.5万".
        for (u, mul) in UNIT_MULTIPLIERS {
            if let Some(rest) = tail.strip_prefix(*u) {
                let rest = rest.trim();
                if rest.is_empty() {
                    return Some(value * mul);
                }
                // Compound unit following the first: `3千万亿` = 3 × 千 ×
                // 万亿 = 3e15. If `rest` is itself a pure-unit string, treat
                // it as an implicit-1 prefix of that unit.
                if let Some(compound) = compound_multiplier(rest) {
                    return Some(value * mul * compound);
                }
                // Otherwise: "1.5万2千" → 1.5 × 10_000 + 2 × 1_000.
                if let Some(extra) = parse_number_body(rest) {
                    return Some(value * mul + extra);
                }
                return None;
            }
        }
        return None;
    }

    // Case B: entirely a pure number ("100.5").
    if PURE_NUMBER.is_match(body) {
        return body.parse().ok();
    }

    // Case C: pure Chinese numeric expression. Reuse char2num.
    char2num(body).ok()
}

/// Try matching a blur-quantifier shape at the front of the body (after
/// stripping case). Covers:
///   * `十几<unit>` → Range(10, 20).
///   * `几十<unit>` → Range(10, 100).  `几百` → 100..1000. etc.
///   * `数十亿` / `数百万` (数 = 3..9).
///   * `X Y <unit>` where both X and Y are 1-9 digit chars (consecutive
///     digits like `八九`, `两三`, `三五` — Range(X*, Y*)).
fn try_blur_quantifier(s: &str, default_unit: &str) -> Option<MoneyInfo> {
    let (sym_case, after_sym) = strip_symbol(s);
    let (prefix_case, after_prefix) = strip_leading_case(after_sym);
    let (suffix_case, body0) = strip_trailing_case(after_prefix);
    // Peel inner 元 if currency was stripped separately (`数十亿元人民币`).
    let body = match strip_trailing_case(body0) {
        (Some(inner), rest) if inner == "元" => rest,
        _ => body0,
    };
    let case = suffix_case
        .or(sym_case)
        .or(prefix_case)
        .unwrap_or_else(|| default_unit.to_string());
    let body = body.trim();

    let digit = |c: char| -> Option<f64> {
        match c {
            '一' => Some(1.0),
            '二' | '两' => Some(2.0),
            '三' => Some(3.0),
            '四' => Some(4.0),
            '五' => Some(5.0),
            '六' => Some(6.0),
            '七' => Some(7.0),
            '八' => Some(8.0),
            '九' => Some(9.0),
            _ => None,
        }
    };

    let chars: Vec<char> = body.chars().collect();
    if chars.is_empty() {
        return None;
    }

    // Detect a leading `(lo, hi)` produced by quantifier patterns.
    let (lo, hi, rest_start) = if chars[0] == '十' && chars.get(1) == Some(&'几') {
        (10.0, 20.0, 2)
    } else if chars[0] == '十' && chars.get(1) == Some(&'多') {
        // `十多` = 10-20 (same as `十几`).
        (10.0, 20.0, 2)
    } else if chars[0] == '几' && chars.get(1) == Some(&'十') {
        // Python convention: 几十 = 10..100 (upper bound is next power).
        (10.0, 100.0, 2)
    } else if chars[0] == '几' && chars.get(1) == Some(&'百') {
        (100.0, 1000.0, 2)
    } else if chars[0] == '几' && chars.get(1) == Some(&'千') {
        (1_000.0, 10_000.0, 2)
    } else if chars[0] == '几' && chars.get(1) == Some(&'万') {
        (10_000.0, 100_000.0, 2)
    } else if chars[0] == '几' && chars.get(1) == Some(&'亿') {
        (1e8, 1e9, 2)
    } else if chars[0] == '数' && chars.get(1) == Some(&'十') {
        // Python convention: `数十` = 10-100 (both bounds power-of-10).
        (10.0, 100.0, 2)
    } else if chars[0] == '数' && chars.get(1) == Some(&'百') {
        (100.0, 1000.0, 2)
    } else if chars[0] == '数' && chars.get(1) == Some(&'千') {
        (1_000.0, 10_000.0, 2)
    } else if chars[0] == '数' && chars.get(1) == Some(&'万') {
        (10_000.0, 100_000.0, 2)
    } else if chars[0] == '数' && chars.get(1) == Some(&'亿') {
        (1e8, 1e9, 2)
    } else if let (Some(a), Some(b)) = (digit(chars[0]), chars.get(1).and_then(|c| digit(*c))) {
        // Two consecutive CN digits: 八九 → Range(8, 9).
        (a, b, 2)
    } else {
        return None;
    };

    // Optional trailing multi-character unit multiplier (e.g. `十几块` has
    // no trailing unit; `几十万块` has `万`). Consume remaining as compound.
    let remaining: String = chars[rest_start..].iter().collect();
    let remaining = remaining.trim();
    let mul = if remaining.is_empty() {
        1.0
    } else {
        compound_multiplier(remaining)?
    };

    Some(MoneyInfo {
        num: lo * mul,
        case,
        definition: "blur",
        end_num: Some(hi * mul),
    })
}

/// Interpret a pure-unit string as a compound multiplier. Examples:
///   `"万亿"` → 1e12, `"千万"` → 1e7, `"百万"` → 1e6, `"千亿"` → 1e11.
fn compound_multiplier(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let mut total = 1.0f64;
    let mut seen_any = false;
    let mut rest = s;
    loop {
        let mut matched = false;
        for (u, mul) in UNIT_MULTIPLIERS {
            if let Some(r) = rest.strip_prefix(*u) {
                total *= mul;
                rest = r;
                matched = true;
                seen_any = true;
                break;
            }
        }
        if !matched {
            break;
        }
    }
    if seen_any && rest.is_empty() {
        Some(total)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Arabic-numeric variants ──────────────────────────────────────

    #[test]
    fn plain_with_yuan() {
        let m = parse_money("100元").unwrap();
        assert_eq!(m.num, 100.0);
        assert_eq!(m.case, "元");
    }

    #[test]
    fn comma_thousands() {
        let m = parse_money("1,000元").unwrap();
        assert_eq!(m.num, 1_000.0);
    }

    #[test]
    fn decimal() {
        let m = parse_money("100.50元").unwrap();
        assert!((m.num - 100.5).abs() < 1e-9);
    }

    #[test]
    fn arabic_with_big_unit() {
        let m = parse_money("1.5万元").unwrap();
        assert!((m.num - 15_000.0).abs() < 1e-9);
        assert_eq!(m.case, "元");
    }

    // ── Chinese numeric variants ─────────────────────────────────────

    #[test]
    fn chinese_simple() {
        let m = parse_money("一百元").unwrap();
        assert_eq!(m.num, 100.0);
    }

    #[test]
    fn chinese_wan() {
        let m = parse_money("三千五百万元").unwrap();
        assert_eq!(m.num, 35_000_000.0);
    }

    // ── Currency cases ───────────────────────────────────────────────

    #[test]
    fn usd_suffix() {
        let m = parse_money("100美元").unwrap();
        assert_eq!(m.case, "美元");
        assert_eq!(m.num, 100.0);
    }

    #[test]
    fn prefix_rmb() {
        let m = parse_money("人民币500元").unwrap();
        assert_eq!(m.num, 500.0);
        assert_eq!(m.case, "元");
    }

    #[test]
    fn default_unit_override() {
        let m = parse_money_with_default("100", "美元").unwrap();
        assert_eq!(m.case, "美元");
    }

    // ── Rejections ───────────────────────────────────────────────────

    #[test]
    fn empty_input_returns_none() {
        assert!(parse_money("").is_none());
    }

    #[test]
    fn nonsense_returns_none() {
        assert!(parse_money("abc xyz").is_none());
    }

    // ── Round 6 additions ───────────────────────────────────────────

    #[test]
    fn currency_symbol_prefix_yen() {
        let m = parse_money("￥100").unwrap();
        assert_eq!(m.num, 100.0);
        assert_eq!(m.case, "元");
    }

    #[test]
    fn currency_symbol_prefix_dollar() {
        let m = parse_money("$100.5").unwrap();
        assert_eq!(m.num, 100.5);
        assert_eq!(m.case, "美元");
    }

    #[test]
    fn yuan_jiao_fen_triple() {
        let m = parse_money("100元5角3分").unwrap();
        assert!((m.num - 100.53).abs() < 1e-9);
        assert_eq!(m.case, "元");
    }

    #[test]
    fn yuan_jiao_only() {
        let m = parse_money("100元5角").unwrap();
        assert!((m.num - 100.5).abs() < 1e-9);
    }

    #[test]
    fn blur_prefix() {
        let m = parse_money("约100元").unwrap();
        assert_eq!(m.definition, "blur");
        assert_eq!(m.num, 100.0);
    }

    #[test]
    fn blur_suffix_zuoyou() {
        let m = parse_money("100元左右").unwrap();
        assert_eq!(m.definition, "blur");
    }

    #[test]
    fn range_with_dash() {
        let m = parse_money("100-200元").unwrap();
        assert_eq!(m.num, 100.0);
        assert_eq!(m.end_num, Some(200.0));
    }

    #[test]
    fn range_chinese() {
        let m = parse_money("100到200元").unwrap();
        assert_eq!(m.num, 100.0);
        assert_eq!(m.end_num, Some(200.0));
    }
}
