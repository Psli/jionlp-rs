//! Time parser — first-stage port of `jionlp/gadget/time_parser.py`.
//!
//! The Python implementation is 4865 lines covering dozens of time-language
//! forms. This first-stage port deliberately covers the three highest-value
//! input shapes (empirically ~50% of real traffic on Chinese-text products):
//!
//! 1. **Absolute dates** — `2024年3月5日`, `2024-03-05`, `2024/3/5`,
//!    `2024年3月`, `2024年`.
//! 2. **Relative days** — `今天` `今日` / `明天` `明日` / `昨天` / `后天` /
//!    `前天` / `大后天` / `大前天`.
//! 3. **Clock times**  — `上午8点` `上午8点30分` / `下午3点` / `晚上10点` /
//!    `凌晨4点`, attachable to a preceding date.
//!
//! Anything beyond this (週期性 / 模糊 / 范围 / 农历 / 节假日 /
//! 古代纪元) returns `None` and should be added in follow-up stages.

use chrono::{Datelike, Duration, Local, NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use once_cell::sync::Lazy;
use regex::Regex;

/// A delta component. `Single(v)` means an exact count; `Range(lo, hi)` means
/// a fuzzy estimate like "两三天" → (2, 3) or "30~90日" → (30, 90).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeltaValue {
    Single(f64),
    Range(f64, f64),
}

/// Matches Python's `time_delta` dict. Each unit is an optional count or
/// range; `zero = true` when the user explicitly said `0 天 / 0 小时` (rare
/// but possible in contracts like "0 工作日内").
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TimeDelta {
    pub year: Option<DeltaValue>,
    pub month: Option<DeltaValue>,
    pub day: Option<DeltaValue>,
    pub hour: Option<DeltaValue>,
    pub minute: Option<DeltaValue>,
    pub second: Option<DeltaValue>,
    pub workday: Option<DeltaValue>,
    pub zero: bool,
}

impl TimeDelta {
    pub fn is_empty(&self) -> bool {
        self.year.is_none()
            && self.month.is_none()
            && self.day.is_none()
            && self.hour.is_none()
            && self.minute.is_none()
            && self.second.is_none()
            && self.workday.is_none()
    }
}

/// Matches Python's `time_period` dict — a `delta` (recurrence cadence like
/// `{week: 1}`) plus an optional `point` (instants within one cycle).
#[derive(Debug, Clone, PartialEq)]
pub struct TimePeriodInfo {
    pub delta: TimeDelta,
    /// Each tuple is a `(start, end)` anchor within one cycle. For
    /// `每周六上午9点到11点` there is one anchor `(Sat 09:00, Sat 11:00)`.
    pub point_time: Vec<(NaiveDateTime, NaiveDateTime)>,
    /// The original sub-phrase that described the point (for debugging /
    /// round-tripping). E.g. `"周六上午9点到11点"`.
    pub point_string: String,
}

/// `TimeInfo` carries the normalized interpretation.
///
/// * `time_type` is one of `"time_point"`, `"time_span"`, `"time_delta"`,
///   `"time_period"` — matching the four shapes Python returns.
/// * For `time_point` / `time_span`, `start` and `end` are inclusive
///   endpoints filled to the implied granularity. For `time_delta` /
///   `time_period` the `start`/`end` fields are zero-valued sentinels and
///   callers should inspect `delta` / `period` instead.
/// * `definition` is `"accurate"` or `"blur"`.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeInfo {
    pub time_type: &'static str,
    pub start: NaiveDateTime,
    pub end: NaiveDateTime,
    pub definition: &'static str,
    /// Populated when `time_type == "time_delta"` or `"time_period"`.
    pub delta: Option<TimeDelta>,
    /// Populated only when `time_type == "time_period"`.
    pub period: Option<TimePeriodInfo>,
}

impl Default for TimeInfo {
    fn default() -> Self {
        let zero = NaiveDate::from_ymd_opt(1970, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        TimeInfo {
            time_type: "time_point",
            start: zero,
            end: zero,
            definition: "accurate",
            delta: None,
            period: None,
        }
    }
}

/// Parse `text` relative to the system's local "now".
pub fn parse_time(text: &str) -> Option<TimeInfo> {
    parse_time_with_ref(text, Local::now().naive_local())
}

/// Parse `text` relative to an explicit `now`. Use this for deterministic
/// tests and for scenarios where the server clock differs from the user's.
pub fn parse_time_with_ref(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    // Publish `now.year()` as the ref year so `parse_year` can correctly
    // disambiguate 2-digit years deep in the dispatch. Restore on exit so
    // nested calls (via try_open_ended_span etc.) see the outer context.
    let _guard = RefYearGuard::new(now.year());

    // Order matters.
    //
    // Stage 5 (fuzzy) first — exact-string matches, cheapest path.
    if let Some(t) = try_fuzzy_time(trimmed, now) {
        return Some(t);
    }
    // Round 17 — exact-string special phrases (现在/全年/今明两天/…).
    if let Some(t) = try_special_phrases(trimmed, now) {
        return Some(t);
    }
    // Round 29 — `一年四季` special phrase.
    if let Some(t) = try_yinian_siji(trimmed) {
        return Some(t);
    }
    // Round 29 — `每周工作日<clock>` weekday-filtered recurrence.
    if let Some(t) = try_recurring_weekday_filtered(trimmed, now) {
        return Some(t);
    }
    // Round 20 — blur-hour phrase (`早上`, `晚上`, `今天下午`, `凌晨`, `午夜`).
    if let Some(t) = try_blur_hour_phrase(trimmed, now) {
        return Some(t);
    }
    // Round 20 — approx prefix (`约9点` / `大概下午3点`).
    if let Some(t) = try_approx_clock(trimmed, now) {
        return Some(t);
    }
    // Round 20 — super-blur HMS (`前两个小时` / `未来三分钟`).
    if let Some(t) = try_super_blur_hms(trimmed, now) {
        return Some(t);
    }
    // Round 20 — recurring HMS / gap / yearly-festival — before try_recurring
    // so exact patterns match first.
    if let Some(t) = try_recurring_gap(trimmed, now) {
        return Some(t);
    }
    if let Some(t) = try_recurring_hms(trimmed, now) {
        return Some(t);
    }
    if let Some(t) = try_recurring_yearly_festival(trimmed, now) {
        return Some(t);
    }
    // Round 20 — open-ended span (`2024年3月之后` / `春节之前`). Must run
    // before try_time_delta so non-delta inner doesn't get stolen.
    if let Some(t) = try_open_ended_span(trimmed, now) {
        return Some(t);
    }
    // Round 17 #20 — `20世纪` / `20世纪二十年代` — must precede named-period.
    if let Some(t) = try_century(trimmed, now) {
        return Some(t);
    }
    // Round 17 #8 — `前两天` / `未来三天` super-blur YMD.
    if let Some(t) = try_super_blur_ymd(trimmed, now) {
        return Some(t);
    }
    // Stage 4 paths — narrow prefixes (`本/上/下` or digit + 后/前).
    if let Some(t) = try_named_period(trimmed, now) {
        return Some(t);
    }
    // Round 17 #18 — `下个月9号` / `本月15日`.
    if let Some(t) = try_limit_month_day(trimmed, now) {
        return Some(t);
    }
    // Python-parity: `10月12日` / `10月12日16时` / `10月12` (bare, no year).
    // Inherits year from `now`. Must come before try_absolute_date since
    // that requires a leading year.
    if let Some(t) = try_bare_month_day(trimmed, now) {
        return Some(t);
    }
    // Python-parity: `12日16时` / `12号15点` — day-with-clock, inherits
    // year + month from `now`. Lets date ranges like `本月10日至12日16时`
    // parse their right-hand side correctly.
    if let Some(t) = try_bare_day_with_clock(trimmed, now) {
        return Some(t);
    }
    // Round 17 #16 — `下个月末` / `本月初`.
    if let Some(t) = try_limit_month_boundary(trimmed, now) {
        return Some(t);
    }
    // Round 17 #41 — `本周一` / `下周五`.
    if let Some(t) = try_named_weekday(trimmed, now) {
        return Some(t);
    }
    // Round 29 #13/#14 — `2021年第1季度初/末` — more specific, runs first.
    if let Some(t) = try_year_solar_season_boundary(trimmed) {
        return Some(t);
    }
    // Python parity — relative year + quarter + boundary.
    if let Some(t) = try_relative_year_quarter_boundary(trimmed, now) {
        return Some(t);
    }
    // Python parity — limit quarter + boundary (`上季度末`, etc.).
    if let Some(t) = try_limit_quarter_boundary(trimmed, now) {
        return Some(t);
    }
    // Round 17 #11 — `2021年第1季度`.
    if let Some(t) = try_year_solar_season(trimmed) {
        return Some(t);
    }
    // Python parity — `一季度` / `首季度` / `Q1季度` bare (inherits year).
    if let Some(t) = try_bare_quarter(trimmed, now) {
        return Some(t);
    }
    // Round 29 #17 — school break (暑假/寒假/春假/秋假) with optional year.
    if let Some(t) = try_school_break(trimmed, now) {
        return Some(t);
    }
    // Round 17 #15 — `2021年初` / `2021年末`.
    if let Some(t) = try_year_blur_boundary(trimmed) {
        return Some(t);
    }
    // Python parity — relative year + `初/末/底/中` boundary.
    if let Some(t) = try_relative_year_blur_boundary(trimmed, now) {
        return Some(t);
    }
    // Python parity — `<M月>第N周` / `<Y年M月>第N周` / relative-month /
    // relative-year versions. Covers 14+ parity-corpus cases.
    if let Some(t) = try_month_week_ordinal(trimmed, now) {
        return Some(t);
    }
    // Python parity — `下周周六` / `上周周三` / `本周星期二` — named
    // week prefix + weekday suffix (both 周/星期/礼拜 variants).
    if let Some(t) = try_named_week_weekday(trimmed, now) {
        return Some(t);
    }
    // Python parity — `<year>?上半年/下半年` / `<year>?伊始`.
    if let Some(t) = try_half_year(trimmed, now) {
        return Some(t);
    }
    // Python parity — `<year>首月` / `<year>末月` / `<year>第N个月`
    // / `<year>前N个月` / `<year>后N个月`.
    if let Some(t) = try_year_ordinal_month(trimmed, now) {
        return Some(t);
    }
    // Python parity — `<year>?<M月>上旬/中旬/下旬` (1-10, 11-20, 21-EOM)
    // and `<year>?<M月>底` (last 5 days).
    if let Some(t) = try_month_xun(trimmed, now) {
        return Some(t);
    }
    // Python parity — `同月D号/日[clock]` / `同年M月D号` — inherits
    // year/month from `now`.
    if let Some(t) = try_same_month_or_year(trimmed, now) {
        return Some(t);
    }
    // Python parity — `<相对年>M月[D日|号][<clock>]` / `<相对年>M月份`.
    // Covers 今年六月, 明年3月份, 去年3月3号, 前年9月2号左右, etc.
    if let Some(t) = try_relative_year_month_day(trimmed, now) {
        return Some(t);
    }
    // Python parity — `从X起[至今|到现在|到今天]` / `X至今` / `X到现在`:
    // open-ended span from concrete `X` to `now`.
    if let Some(t) = try_span_to_now(trimmed, now) {
        return Some(t);
    }
    // Python parity — concrete date/time + 左右 / 前后 / 附近 suffix.
    // Strip the approx modifier, reparse, flag definition as `blur`.
    if let Some(t) = try_approx_modifier(trimmed, now) {
        return Some(t);
    }
    // Python parity — delta + 之后/之前/以前/以后 where delta carries a
    // range (`几十年之后`, `二十几年前`). Produces a time_span centered
    // on now ± [low, high] years/months/etc.
    if let Some(t) = try_delta_range_suffix(trimmed, now) {
        return Some(t);
    }
    // Round 29 #48 — `明年第10周` — limit year + week.
    if let Some(t) = try_limit_year_week(trimmed, now) {
        return Some(t);
    }
    // Round 17 #47 — `2021年第5周`.
    if let Some(t) = try_year_week(trimmed) {
        return Some(t);
    }
    // Round 17 #43-46 — `2021年10月第二个周一` / `M月第一个周五`.
    if let Some(t) = try_nth_weekday_in_month(trimmed, now) {
        return Some(t);
    }
    // Round 17 #52/53 — `第365天` / `2025年第一天`.
    if let Some(t) = try_year_day_ordinal(trimmed, now) {
        return Some(t);
    }
    // Round 29 #54 — `第N年` year ordinal.
    if let Some(t) = try_year_ordinal(trimmed, now) {
        return Some(t);
    }
    // Round 29 #10 — `今年/明年/去年 (的) N个?(月|季度)`.
    if let Some(t) = try_limit_year_span_month(trimmed, now) {
        return Some(t);
    }
    // Round 19 — span-shaped delta prefixes (`未来三天` / `最近一周` /
    // `再过五分钟` / `过3个小时`). Must run before try_time_delta so that
    // `未来三天` isn't eaten by try_super_blur_ymd first. Actually ordered:
    // super_blur handles 未来/过去/近 already; this parser is backup for the
    // 再过/过 forms.
    if let Some(t) = try_delta_to_span(trimmed, now) {
        return Some(t);
    }
    // Round 19 — `5分钟内` / `10秒来` → span covering last N <unit>.
    if let Some(t) = try_delta_inner_span(trimmed, now) {
        return Some(t);
    }
    // Round 19 — `1个工作日前` / `3个工作日后`. Before try_time_delta so the
    // 工作日 suffix isn't mis-matched as 日 only.
    if let Some(t) = try_workday_delta_point(trimmed, now) {
        return Some(t);
    }
    // Round 32 #70 — nested delta `N<unit1>的M<unit2>(后|前)`.
    if let Some(t) = try_nested_delta_point(trimmed, now) {
        return Some(t);
    }
    if let Some(t) = try_time_delta(trimmed, now) {
        return Some(t);
    }
    // Pattern #90 — `30~90日` / `2到5年` — pure time_delta with range.
    if let Some(t) = try_delta_range(trimmed, now) {
        return Some(t);
    }
    // Pattern #9 — `2021年前两个季度` / `2021年最后三个月` — year+span.
    if let Some(t) = try_year_span_month_or_quarter(trimmed, now) {
        return Some(t);
    }

    // Stage 3 (recurring) must come first among mid-prefix parsers: "每周一"
    // starts with 每 and wouldn't match anything else, but adding it early
    // lets us reject early before running the more expensive date parsers.
    if let Some(t) = try_recurring(trimmed, now) {
        return Some(t);
    }
    // Stage 3 (timespan) — clock-range on implied or explicit date, e.g.
    // "8点到12点" / "明天下午3点到5点". Must run before date_range so its
    // "到" separator isn't stolen.
    if let Some(t) = try_clock_range(trimmed, now) {
        return Some(t);
    }
    // Range must come before date parser (it looks for a separator that
    // would otherwise be eaten by try_absolute_date).
    if let Some(t) = try_date_range(trimmed, now) {
        return Some(t);
    }
    // Round 18 — limit-year + festival: `今年儿童节` / `明年春节` /
    // `去年母亲节`. Runs before try_fixed_holiday so the prefix is stripped.
    if let Some(t) = try_limit_year_festival(trimmed, now) {
        return Some(t);
    }
    if let Some(t) = try_fixed_holiday(trimmed, now) {
        return Some(t);
    }
    // Round 29 #7 — `M月<d1>日、<d2>日、…`.
    if let Some(t) = try_enum_days(trimmed, now) {
        return Some(t);
    }
    // Round 18 — solar terms (24 节气): `2024年清明` / `立春` / `今年冬至`.
    if let Some(t) = try_solar_term(trimmed, now) {
        return Some(t);
    }
    // Round 18 — seasons: `2024年春天` / `去年夏季`.
    if let Some(t) = try_season(trimmed, now) {
        return Some(t);
    }
    // Round 18 — lunar date: `农历九月十二` / `2012年农历正月十九` / `腊月二十八`.
    if let Some(t) = try_lunar_date(trimmed, now) {
        return Some(t);
    }
    if let Some(t) = try_absolute_date(trimmed) {
        return Some(t);
    }
    // Round 19 — bare `3年` / `两个月` / `一万个小时` → time_delta dict. Runs
    // AFTER try_absolute_date so `2024年` doesn't get mis-read as 2024-year
    // delta.
    if let Some(t) = try_pure_delta(trimmed) {
        return Some(t);
    }
    // Round 19 — `5分钟内` / `48小时之前` / `3个月之后` — span-shaped delta
    // forms. Placed here so `之后/之前` suffix doesn't steal `三天之后` from
    // try_time_delta's time_point output.
    if let Some(t) = try_delta_inner_span(trimmed, now) {
        return Some(t);
    }
    if let Some(t) = try_delta_open_ended_span_subday(trimmed, now) {
        return Some(t);
    }
    // Round 17 #2 — `20210901` 8-digit. After absolute_date so 2024-3-5 style
    // wins, but before clock_only (4-digit hours).
    if let Some(t) = try_eight_digit_ymd(trimmed) {
        return Some(t);
    }
    if let Some(t) = try_relative_day(trimmed, now) {
        return Some(t);
    }
    // Round 17 #40 — standalone `周一` / `星期五`.
    if let Some(t) = try_standalone_weekday(trimmed, now) {
        return Some(t);
    }
    if let Some(t) = try_clock_only(trimmed, now.date()) {
        return Some(t);
    }
    None
}

// ───────────────────────── absolute date ────────────────────────────────────

static YMD_CN: Lazy<Regex> = Lazy::new(|| {
    // "2024年3月5日" or "2024年3月5号" (with optional clock tail).
    Regex::new(r"^(\d{2,4})\s*年\s*(?:(\d{1,2})\s*月\s*(?:(\d{1,2})\s*[日号])?)?\s*(.*)$").unwrap()
});

/// Same as YMD_CN but accepts a leading run of Chinese numeral digits
/// before 年. Group 1 captures the raw Chinese digits (to be normalized
/// via `parse_chinese_year_digits`), groups 2..4 mirror YMD_CN.
static YMD_CN_CHINESE: Lazy<Regex> = Lazy::new(|| {
    // Allow 〇零一二三四五六七八九十 and 两 for year digits, and accept
    // either Arabic or Chinese numerals for month and day. The month
    // regex allows up to 3 chars (`十一`, `十二`, `二十`), day up to
    // 4 chars (`二十八`, `三十一`).
    Regex::new(
        r"^([零〇一二三四五六七八九十两]+)\s*年\s*(?:(\d{1,2}|[零〇一二三四五六七八九十]{1,3})\s*月\s*(?:(\d{1,2}|[零〇一二三四五六七八九十]{1,4})\s*[日号])?)?\s*(.*)$",
    )
    .unwrap()
});

/// Convert a Chinese-numeral year string to a 4-digit integer. Handles:
///   "零三" → 2003  (00-69 expand to 2000s, per Python convention)
///   "九八" → 1998  (70-99 expand to 1900s)
///   "二零二四" → 2024  (4-char per-digit)
///   "二〇〇三" → 2003
///   "二零〇三" → 2003
/// Returns `None` for anything else.
fn parse_chinese_year_digits(s: &str) -> Option<i32> {
    let mut digits = String::new();
    for c in s.chars() {
        let d = match c {
            '零' | '〇' => '0',
            '一' => '1',
            '二' | '两' => '2',
            '三' => '3',
            '四' => '4',
            '五' => '5',
            '六' => '6',
            '七' => '7',
            '八' => '8',
            '九' => '9',
            _ => return None,
        };
        digits.push(d);
    }
    if digits.is_empty() {
        return None;
    }
    // Route through `parse_year` so 2-digit years inherit the same
    // ref-year-aware disambiguation (`三三年` at ref 2021 → 1933 not 2033).
    if digits.len() == 2 {
        parse_year(&digits)
    } else if digits.len() == 4 {
        digits.parse().ok()
    } else {
        None
    }
}

static YMD_DASH: Lazy<Regex> = Lazy::new(|| {
    // Accept -, /, and . as YMD separators. Mixed forms like `1994.01-19`
    // are supported by alternating (anything in -, /, .) around each number.
    // Tail (group 4) must look like a clock — whitespace-separated free
    // text, or a compact digit-glued clock (e.g. `2018-12-1209:03`,
    // `2021-09-12-11：23`) containing `:` or `：` or a Chinese clock
    // marker (点/时/分/秒). This prevents `1999.08-2002.02` from being
    // (wrongly) absorbed as YMD + spurious trailing.
    // Accept ASCII space, -, /, . and `·` as YMD separators. Tail (group 4)
    // must look like a clock, so `2022 11 23` parses but `2022 11 abc`
    // doesn't accidentally succeed.
    Regex::new(r"^(\d{4})[\-/\.\s·](\d{1,2})[\-/\.\s·](\d{1,2})(\s+.+|[\-\s]*\d+\s*[:：点时].*|[\-\s]*\d{2}:\d{2}.*)?$").unwrap()
});

/// `YYYY.MM` / `YYYY-MM` / `YYYY/MM` — year+month only (no day). Returns
/// a `time_span` covering the whole month. Used by `try_absolute_date` as
/// a secondary pattern when YMD_DASH fails; needed so that
/// `1999.08-2002.02` parses as a range of month spans.
static YM_DASH: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(\d{4})[\-/\.](\d{1,2})$").unwrap());

/// Bare 4-digit year `2018` → full-year time_span.
static YEAR_ONLY: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(\d{4})$").unwrap());

fn try_absolute_date(text: &str) -> Option<TimeInfo> {
    if let Some(caps) = YMD_CN.captures(text) {
        let year = parse_year(caps.get(1)?.as_str())?;
        let month = caps.get(2).and_then(|m| m.as_str().parse::<u32>().ok());
        let day = caps.get(3).and_then(|m| m.as_str().parse::<u32>().ok());
        let tail = caps.get(4).map(|m| m.as_str().trim()).unwrap_or("");

        let (start, end) = date_range(year, month, day)?;
        return apply_optional_clock(start, end, tail);
    }
    if let Some(caps) = YMD_CN_CHINESE.captures(text) {
        let year = parse_chinese_year_digits(caps.get(1)?.as_str())?;
        let month = caps
            .get(2)
            .and_then(|m| {
                let s = m.as_str();
                s.parse::<u32>().ok().or_else(|| cn_int(s))
            })
            .filter(|&m| (1..=12).contains(&m));
        let day = caps
            .get(3)
            .and_then(|m| {
                let s = m.as_str();
                s.parse::<u32>().ok().or_else(|| cn_int(s))
            })
            .filter(|&d| (1..=31).contains(&d));
        let tail = caps.get(4).map(|m| m.as_str().trim()).unwrap_or("");

        let (start, end) = date_range(year, month, day)?;
        return apply_optional_clock(start, end, tail);
    }
    if let Some(caps) = YMD_DASH.captures(text) {
        let year = caps.get(1)?.as_str().parse::<i32>().ok()?;
        let month = caps.get(2)?.as_str().parse::<u32>().ok()?;
        let day = caps.get(3)?.as_str().parse::<u32>().ok()?;
        let tail = caps.get(4).map(|m| m.as_str().trim()).unwrap_or("");

        let (start, end) = date_range(year, Some(month), Some(day))?;
        return apply_optional_clock(start, end, tail);
    }
    if let Some(caps) = YEAR_ONLY.captures(text) {
        let year = caps.get(1)?.as_str().parse::<i32>().ok()?;
        // Require plausible modern year window to avoid mis-matching
        // 8-digit sequences (those are handled by try_eight_digit_ymd).
        if (1900..=2100).contains(&year) {
            let (start, end) = date_range(year, None, None)?;
            return Some(TimeInfo {
                time_type: "time_span",
                start,
                end,
                definition: "accurate",
                ..Default::default()
            });
        }
    }
    if let Some(caps) = YM_DASH.captures(text) {
        let year = caps.get(1)?.as_str().parse::<i32>().ok()?;
        let month = caps.get(2)?.as_str().parse::<u32>().ok()?;
        if (1900..=2100).contains(&year) && (1..=12).contains(&month) {
            let (start, end) = date_range(year, Some(month), None)?;
            return Some(TimeInfo {
                time_type: "time_span",
                start,
                end,
                definition: "accurate",
                ..Default::default()
            });
        }
    }
    None
}

/// Expand 2-digit years the same way the Python library does: 70..99 →
/// 1900s, 00..69 → 2000s. Fully-typed 4-digit years pass through.
/// Ref year used by `parse_year` to disambiguate 2-digit years. Set by
/// `parse_time_with_ref` at entry so patterns deep in the dispatch can
/// resolve `三三年` differently in 2021 vs 2099.
std::thread_local! {
    static REF_YEAR: std::cell::Cell<i32> = const { std::cell::Cell::new(2025) };
}

fn set_ref_year(y: i32) -> i32 {
    REF_YEAR.with(|c| {
        let prev = c.get();
        c.set(y);
        prev
    })
}

fn current_ref_year() -> i32 {
    REF_YEAR.with(|c| c.get())
}

/// RAII guard that restores the previous `REF_YEAR` on drop. Used at
/// `parse_time_with_ref` entry so nested recursive parses see the
/// outer caller's context.
struct RefYearGuard {
    prev: i32,
}

impl RefYearGuard {
    fn new(y: i32) -> Self {
        let prev = set_ref_year(y);
        RefYearGuard { prev }
    }
}

impl Drop for RefYearGuard {
    fn drop(&mut self) {
        set_ref_year(self.prev);
    }
}

/// Expand 2-digit years using Python's `_year_completion` convention:
///   - If the reference year starts with 19 (or older), use that century.
///   - If the reference starts with 20, compare the 2-digit string:
///     if `n > ref_last2 + 10`, drop to 19xx; else stay in 20xx.
/// 4-digit years pass through unchanged.
fn parse_year(s: &str) -> Option<i32> {
    let n: i32 = s.parse().ok()?;
    if s.len() != 2 {
        return Some(n);
    }
    let ref_year = current_ref_year();
    if ref_year < 1900 || ref_year >= 2100 {
        // Fallback to static 70/00 cutoff outside the supported range.
        return Some(if n >= 70 { 1900 + n } else { 2000 + n });
    }
    let ref_century_prefix = ref_year / 100; // 19 or 20
    if ref_century_prefix == 19 {
        return Some(1900 + n);
    }
    // ref is 20xx
    let ref_last2 = ref_year % 100;
    if n > ref_last2 + 10 {
        Some(1900 + n)
    } else {
        Some(2000 + n)
    }
}

/// Produce the [start, end] range for a given date granularity.
fn date_range(
    year: i32,
    month: Option<u32>,
    day: Option<u32>,
) -> Option<(NaiveDateTime, NaiveDateTime)> {
    match (month, day) {
        (Some(m), Some(d)) => {
            let s = NaiveDate::from_ymd_opt(year, m, d)?.and_hms_opt(0, 0, 0)?;
            let e = NaiveDate::from_ymd_opt(year, m, d)?.and_hms_opt(23, 59, 59)?;
            Some((s, e))
        }
        (Some(m), None) => {
            let s = NaiveDate::from_ymd_opt(year, m, 1)?.and_hms_opt(0, 0, 0)?;
            // Last day of month = first of next - 1 day.
            let next_year = if m == 12 { year + 1 } else { year };
            let next_month = if m == 12 { 1 } else { m + 1 };
            let e_date = NaiveDate::from_ymd_opt(next_year, next_month, 1)?.pred_opt()?;
            let e = e_date.and_hms_opt(23, 59, 59)?;
            Some((s, e))
        }
        (None, None) => {
            let s = NaiveDate::from_ymd_opt(year, 1, 1)?.and_hms_opt(0, 0, 0)?;
            let e = NaiveDate::from_ymd_opt(year, 12, 31)?.and_hms_opt(23, 59, 59)?;
            Some((s, e))
        }
        _ => None,
    }
}

// ───────────────────────── relative days ────────────────────────────────────

fn try_relative_day(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    // Order: longest prefix first so `大大后天` beats `大后天`.
    const DAYS: &[(&str, i64)] = &[
        ("大大前天", -4),
        ("大大后天", 4),
        ("大前天", -3),
        ("大后天", 3),
        ("前天", -2),
        ("后天", 2),
        ("昨天", -1),
        ("昨日", -1),
        ("昨晚", -1), // 昨晚 = last night — attach default evening period later.
        ("明天", 1),
        ("明日", 1),
        ("明晚", 1),
        ("今天", 0),
        ("今日", 0),
        ("今晚", 0),
        ("本日", 0),
        ("当日", 0),
        ("当天", 0),
        ("当晚", 0),
    ];

    for (kw, offset) in DAYS {
        if let Some(rest) = text.strip_prefix(*kw) {
            let rest = rest.trim_start();
            let base = now.date() + Duration::days(*offset);
            let start = base.and_hms_opt(0, 0, 0)?;
            let end = base.and_hms_opt(23, 59, 59)?;
            // Evening prefix (昨晚/今晚/当晚/明晚) applies an implicit
            // `晚上` qualifier to the clock so `8时` becomes 20:00.
            let implies_evening = matches!(*kw, "昨晚" | "今晚" | "当晚" | "明晚");
            let rest = rest.to_string();
            let rest = if implies_evening
                && !rest.is_empty()
                && !rest.starts_with("晚上")
                && !rest.starts_with("晚")
                && !rest.starts_with("上午")
                && !rest.starts_with("下午")
                && !rest.starts_with("中午")
                && !rest.starts_with("凌晨")
                && !rest.starts_with("傍晚")
            {
                format!("晚上{}", rest)
            } else {
                rest
            };
            return apply_optional_clock(start, end, &rest);
        }
    }
    None
}

// ───────────────────────── clock attachment ─────────────────────────────────

static CLOCK: Lazy<Regex> = Lazy::new(|| {
    // Hour, optional minute (after `点`/`时`/`:`), optional seconds — accepts
    // both Chinese `N秒` and ISO-style `:SS` tail (e.g. `09:36:46`).
    Regex::new(
        r"^(凌晨|早[上晨]?|上午|中午|下午|午后|晚上|傍晚|夜里|夜间)?\s*(\d{1,2})\s*(?:[点时:](\d{1,2})?\s*(?:分)?\s*(?::(\d{1,2})|(\d{1,2})\s*秒)?)?",
    )
    .unwrap()
});

/// Given a day-range and an optional trailing clock expression, narrow the
/// range to a single second when the clock is present; otherwise return
/// the original range.
fn apply_optional_clock(start: NaiveDateTime, end: NaiveDateTime, tail: &str) -> Option<TimeInfo> {
    if tail.is_empty() {
        return Some(TimeInfo {
            time_type: "time_point",
            start,
            end,
            definition: "accurate",
            ..Default::default()
        });
    }

    // Tolerate a leading dash/slash before the clock (e.g.
    // `2021-09-12-11：23` — tail `-11：23`). Also strip fullwidth colon
    // variants before parse_clock sees them.
    let tail_clean = tail.trim_start_matches(['-', '/', '.']).replace('：', ":");
    let clock = parse_clock(&tail_clean)?;
    let dt = start.date().and_time(clock);
    // Precision-aware end: seconds → same instant; minute → :59s;
    // hour → :59m:59s. Run on the normalized tail so `：` counts.
    let end_dt = if tail_clean.contains('秒') || tail_clean.matches(':').count() >= 2 {
        dt
    } else if tail_clean.contains('分') || tail_clean.contains(':') {
        dt.with_second(59).unwrap_or(dt)
    } else {
        dt.with_minute(59)
            .and_then(|t| t.with_second(59))
            .unwrap_or(dt)
    };
    Some(TimeInfo {
        time_type: "time_point",
        start: dt,
        end: end_dt,
        definition: "accurate",
        ..Default::default()
    })
}

/// Replace leading Chinese numerals before `点/时` and between `点/时`
/// and `分` with Arabic digits. Anything else passes through.
fn cn_normalize_clock(s: &str) -> String {
    static RE_HOUR: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"([零〇一二两三四五六七八九十]+)([点时])").unwrap());
    static RE_MIN: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"([点时])([零〇一二两三四五六七八九十]+)(分)").unwrap());
    static RE_SEC: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(分)([零〇一二两三四五六七八九十]+)(秒)").unwrap());
    fn sub_once(input: &str, re: &Regex) -> String {
        let mut out = String::new();
        let mut last = 0;
        for caps in re.captures_iter(input) {
            let m = caps.get(0).unwrap();
            out.push_str(&input[last..m.start()]);
            // For RE_HOUR: group1=digits, group2=unit char.
            // For RE_MIN/SEC: group1=left-marker, group2=digits, group3=right-unit.
            let groups = caps.len();
            if groups == 3 {
                // RE_HOUR
                let n_str = caps.get(1).map(|m| m.as_str()).unwrap_or("");
                let unit = caps.get(2).map(|m| m.as_str()).unwrap_or("");
                if let Some(n) = cn_int(n_str) {
                    out.push_str(&n.to_string());
                    out.push_str(unit);
                } else {
                    out.push_str(m.as_str());
                }
            } else {
                let lm = caps.get(1).map(|m| m.as_str()).unwrap_or("");
                let n_str = caps.get(2).map(|m| m.as_str()).unwrap_or("");
                let rm = caps.get(3).map(|m| m.as_str()).unwrap_or("");
                if let Some(n) = cn_int(n_str) {
                    out.push_str(lm);
                    out.push_str(&n.to_string());
                    out.push_str(rm);
                } else {
                    out.push_str(m.as_str());
                }
            }
            last = m.end();
        }
        out.push_str(&input[last..]);
        out
    }
    let a = sub_once(s, &RE_HOUR);
    let b = sub_once(&a, &RE_MIN);
    sub_once(&b, &RE_SEC)
}

/// Parse a clock expression. Accepts:
///   "8点", "8点30分", "8点30分15秒"
///   "8点半"   = 8:30    "8点一刻" = 8:15     "8点三刻" = 8:45
///   "上午8点", "下午3点", "晚上10点", "凌晨4点", "中午12点"
///   "08:30", "08:30:15"
fn parse_clock(s: &str) -> Option<NaiveTime> {
    let s = s.trim();
    // Normalize Chinese numeral hours/minutes to Arabic so CLOCK regex
    // (which only matches \d) can locate them. Covers `十一点半`,
    // `十时三十分`, `八时三十五分` etc.
    let normalized = cn_normalize_clock(s);
    let s = normalized.as_str();

    // Strip Chinese "fractional hour" suffixes first — they carry an
    // implicit minute count and must be handled before the digit regex.
    let (s, implicit_min): (&str, Option<u32>) = if let Some(rest) = s.strip_suffix("点半") {
        (rest, Some(30))
    } else if let Some(rest) = s.strip_suffix("时半") {
        (rest, Some(30))
    } else if let Some(rest) = s.strip_suffix("点一刻") {
        (rest, Some(15))
    } else if let Some(rest) = s
        .strip_suffix("点二刻")
        .or_else(|| s.strip_suffix("点两刻"))
    {
        (rest, Some(30))
    } else if let Some(rest) = s.strip_suffix("点三刻") {
        (rest, Some(45))
    } else {
        (s, None)
    };

    // Tack a synthetic "点" back on when we stripped a suffix, so the
    // digit regex can still locate the hour.
    let probe: String;
    let target: &str = if implicit_min.is_some() {
        probe = format!("{}点", s);
        probe.as_str()
    } else {
        s
    };

    let caps = CLOCK.captures(target)?;
    let period = caps.get(1).map(|m| m.as_str().to_string());
    let hour: u32 = caps.get(2)?.as_str().parse().ok()?;

    let minute: u32 = if let Some(m) = implicit_min {
        m
    } else {
        caps.get(3)
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0)
    };
    let second: u32 = if implicit_min.is_some() {
        0
    } else {
        // Group 4 = `:SS` variant, group 5 = `N秒` variant.
        caps.get(4)
            .or_else(|| caps.get(5))
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0)
    };

    let adjusted_hour = normalize_hour_by_period(hour, period.as_deref());
    NaiveTime::from_hms_opt(adjusted_hour, minute, second)
}

/// Map Chinese period qualifiers to 24-hour time.
///
/// Rules:
///   * 凌晨 / 早上 / 早晨 / 上午  → 0-11 stays, 12 becomes 0
///   * 中午                       → 12 stays as 12
///   * 下午 / 午后 / 晚上 / 傍晚 / 夜里 / 夜间 → hour < 12 → +12; hour ==12 stays
fn normalize_hour_by_period(hour: u32, period: Option<&str>) -> u32 {
    if hour >= 24 {
        return hour; // let caller reject via `from_hms_opt`
    }
    match period {
        Some("凌晨") | Some("早上") | Some("早晨") | Some("早") | Some("上午") => {
            if hour == 12 {
                0
            } else {
                hour
            }
        }
        Some("中午") => hour, // 12:30 stays at 12:30
        Some("下午") | Some("午后") | Some("晚上") | Some("傍晚") | Some("夜里") | Some("夜间") => {
            if hour < 12 {
                hour + 12
            } else {
                hour
            }
        }
        _ => hour,
    }
}

// ───────────────────────── bare clock ───────────────────────────────────────

// ───────────────────────── fixed holidays ──────────────────────────────────

/// Gregorian-fixed Chinese public holidays. Entries are (name, month, day).
/// 春节/中秋/端午 depend on 农历 and are NOT handled in Stage 2 — adding
/// them requires a lunar calendar table. Tracked in PLAN.md as Stage 3.
const FIXED_HOLIDAYS: &[(&str, u32, u32)] = &[
    ("元旦", 1, 1),
    ("三八妇女节", 3, 8),
    ("妇女节", 3, 8),
    ("植树节", 3, 12),
    ("清明节", 4, 5), // 公历 4/5 附近,忽略 ±1 天精度
    ("清明", 4, 5),
    ("劳动节", 5, 1),
    ("五一", 5, 1),
    ("青年节", 5, 4),
    ("五四", 5, 4),
    ("儿童节", 6, 1),
    ("建党节", 7, 1),
    ("建军节", 8, 1),
    ("教师节", 9, 10),
    ("国庆节", 10, 1),
    ("国庆", 10, 1),
    ("万圣节", 10, 31),
    ("光棍节", 11, 11),
    ("双十一", 11, 11),
    ("双11", 11, 11),
    ("圣诞节", 12, 25),
    ("圣诞", 12, 25),
];

/// Holidays that fall on the Nth-weekday-of-month (vs a fixed date).
/// Tuple is (name, month, weekday_0_indexed_from_monday, nth_occurrence_1_indexed).
const NTH_WEEKDAY_HOLIDAYS: &[(&str, u32, u32, u32)] = &[
    ("感恩节", 11, 3, 4),     // 4th Thursday of November (Monday=0 → Thursday=3)
    ("母亲节", 5, 6, 2),      // 2nd Sunday of May (Sunday=6 under num_days_from_monday)
    ("父亲节", 6, 6, 3),      // 3rd Sunday of June
    ("黑色星期五", 11, 4, 4), // Day after Thanksgiving → 4th Friday of November
];

fn try_fixed_holiday(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    // Strip an optional year prefix like "2024年" or "零三年" so
    // "2024年国庆节" / "零三年元宵节" works.
    let (year, rest) = {
        // Try ASCII-digit year first (2024年...).
        let ascii = YMD_CN
            .captures(text)
            .filter(|c| c.get(2).is_none())
            .and_then(|c| {
                let y = parse_year(c.get(1)?.as_str())?;
                Some((y, c.get(4).map(|m| m.as_str().trim()).unwrap_or("")))
            });
        let cn = YMD_CN_CHINESE
            .captures(text)
            .filter(|c| c.get(2).is_none())
            .and_then(|c| {
                let y = parse_chinese_year_digits(c.get(1)?.as_str())?;
                Some((y, c.get(4).map(|m| m.as_str().trim()).unwrap_or("")))
            });
        ascii.or(cn).unwrap_or((now.date().year(), text))
    };

    // Match against the longest holiday name first to avoid "清明" stealing
    // priority from "清明节". Three sources: fixed-Gregorian, lunar, and
    // nth-weekday-of-month. Pick longest prefix wins.
    let fixed = FIXED_HOLIDAYS
        .iter()
        .filter(|(name, _, _)| rest.starts_with(name))
        .max_by_key(|(name, _, _)| name.chars().count())
        .map(|&(n, m, d)| (n, m, d));
    let lunar = LUNAR_HOLIDAYS
        .iter()
        .filter(|(name, _)| rest.starts_with(name))
        .max_by_key(|(name, _)| name.chars().count())
        .and_then(|&(name, holiday)| lunar_holiday_date(year, holiday).map(|(m, d)| (name, m, d)));
    let nth = NTH_WEEKDAY_HOLIDAYS
        .iter()
        .filter(|(name, _, _, _)| rest.starts_with(name))
        .max_by_key(|(name, _, _, _)| name.chars().count())
        .and_then(|&(name, month, wday, nth)| {
            nth_weekday_of_month(year, month, wday, nth).map(|d| (name, month, d.day()))
        });

    // Pick whichever matched with the longest prefix.
    let cands: [Option<(&str, u32, u32)>; 3] = [fixed, lunar, nth];
    let (matched_name, month, day) = cands
        .iter()
        .flatten()
        .max_by_key(|(name, _, _)| name.chars().count())
        .copied()?;

    let tail = rest[matched_name.len()..].trim();
    let (start, end) = date_range(year, Some(month), Some(day))?;
    apply_optional_clock(start, end, tail)
}

// ───────────────────────── lunar-holiday table ──────────────────────────────

/// Chinese lunar-calendar holidays with Gregorian dates hard-coded for
/// 2020-2035 (16 years). Fuller implementation would embed a lunar→
/// Gregorian converter table; the hardcoded approach keeps the crate
/// dependency-free and covers the forseeable production use.

#[derive(Debug, Clone, Copy, PartialEq)]
enum LunarHoliday {
    ChineseNewYear,  // 春节 / 大年初一
    LanternFestival, // 元宵节
    DragonBoat,      // 端午节
    QiXi,            // 七夕
    MidAutumn,       // 中秋节
    ChongYang,       // 重阳
    NewYearEve,      // 除夕
}

const LUNAR_HOLIDAYS: &[(&str, LunarHoliday)] = &[
    ("春节", LunarHoliday::ChineseNewYear),
    ("大年初一", LunarHoliday::ChineseNewYear),
    ("元宵节", LunarHoliday::LanternFestival),
    ("元宵", LunarHoliday::LanternFestival),
    ("端午节", LunarHoliday::DragonBoat),
    ("端午", LunarHoliday::DragonBoat),
    ("七夕节", LunarHoliday::QiXi),
    ("七夕", LunarHoliday::QiXi),
    ("中秋节", LunarHoliday::MidAutumn),
    ("中秋", LunarHoliday::MidAutumn),
    ("重阳节", LunarHoliday::ChongYang),
    ("重阳", LunarHoliday::ChongYang),
    ("除夕", LunarHoliday::NewYearEve),
];

/// Look up the Gregorian (month, day) for a given lunar holiday in a given
/// year. Powered by [`crate::gadget::lunar_solar::lunar_to_solar`],
/// covering 1900-2100. Returns `None` only when the target date doesn't
/// exist (e.g. some years have a short 12th month with only 29 days, so
/// 除夕 falls on day 29 not 30).
fn lunar_holiday_date(year: i32, holiday: LunarHoliday) -> Option<(u32, u32)> {
    use crate::gadget::lunar_solar::lunar_to_solar;

    let (lunar_month, lunar_day_strategy) = match holiday {
        LunarHoliday::ChineseNewYear => (1u32, HolidayDay::Fixed(1)),
        LunarHoliday::LanternFestival => (1, HolidayDay::Fixed(15)),
        LunarHoliday::DragonBoat => (5, HolidayDay::Fixed(5)),
        LunarHoliday::QiXi => (7, HolidayDay::Fixed(7)),
        LunarHoliday::MidAutumn => (8, HolidayDay::Fixed(15)),
        LunarHoliday::ChongYang => (9, HolidayDay::Fixed(9)),
        // 除夕 = last day of previous lunar year's 12th month.
        // Convention in Python jionlp: 除夕 of *year* is the day before the
        // Gregorian New Year's Day for *year*. We interpret "2024 除夕" as
        // "the lunar new year's eve that falls *in* 2024" — i.e. the day
        // before 2024's Chinese New Year. That means decoding lunar 2023's
        // last-day.
        LunarHoliday::NewYearEve => {
            let prev = year - 1;
            let last_day = lunar_month_length(prev, 12)?;
            let d = lunar_to_solar(prev, 12, last_day, false)?;
            return Some((d.month(), d.day()));
        }
    };

    let HolidayDay::Fixed(day) = lunar_day_strategy;
    let g = lunar_to_solar(year, lunar_month, day, false)?;
    Some((g.month(), g.day()))
}

enum HolidayDay {
    Fixed(u32),
}

/// Compute the Nth occurrence of a given weekday within a calendar month.
/// Weekday is 0-indexed from Monday (chrono `num_days_from_monday`).
/// `nth` is 1-based. Returns None if the month doesn't contain that many
/// of the requested weekday (rare — only at `nth == 5` for short months).
fn nth_weekday_of_month(year: i32, month: u32, weekday: u32, nth: u32) -> Option<NaiveDate> {
    let first = NaiveDate::from_ymd_opt(year, month, 1)?;
    let first_wday = first.weekday().num_days_from_monday();
    // Days to add so the 1st-of-month + delta lands on the requested weekday.
    let delta_to_first_match = (weekday + 7 - first_wday) % 7;
    let day = 1 + delta_to_first_match + (nth - 1) * 7;
    NaiveDate::from_ymd_opt(year, month, day)
}

/// Local alias for the public [`crate::gadget::lunar_solar::lunar_month_length`].
fn lunar_month_length(year: i32, month: u32) -> Option<u32> {
    crate::gadget::lunar_solar::lunar_month_length(year, month)
}

// ───────────────────────── date range ──────────────────────────────────────

fn try_date_range(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    // Stage 2 handles: "A到B" / "A至B" / "A-B" / "A~B" / "A—B"
    // where A is a full date and B can be full-date or "just day" form that
    // inherits A's year & month. E.g. "2024年3月5日到8日", "从2018年12月九号
    // 到十五号", "从2024年到2025年".
    let text = text
        .strip_prefix('从')
        .or_else(|| text.strip_prefix('自'))
        .unwrap_or(text);
    // Python-parity: also strip trailing 止/止at the end (`自X至Y止`).
    let text = text.strip_suffix('止').unwrap_or(text);
    // And the `起` marker between A and B (`自X起至Y[止]`).
    // Handled uniformly via split — not a prefix/suffix operation.

    // If the whole text parses as a single Y.M-D date (e.g. `1994.01-19`),
    // prefer that over range splitting.
    if YMD_DASH.is_match(text) {
        return None;
    }

    // Longest first so `——` / `--` beat `—` / `-` and don't leave a
    // stray dash on the right side.
    const SEPS: &[&str] = &["——", "--", "到", "至", "—", "-", "~", "～"];

    let mut best: Option<(usize, usize)> = None; // (byte_pos, sep_len)
    for sep in SEPS {
        if let Some(idx) = text.find(sep) {
            if best.map_or(true, |(b, _)| idx > b) {
                best = Some((idx, sep.len()));
            }
        }
    }
    let (idx, len) = best?;
    let left = text[..idx].trim();
    let right = text[idx + len..].trim();
    // Python-parity: trim trailing `起` from LHS (`自X起至Y止`).
    let left = left.strip_suffix('起').unwrap_or(left).trim();
    if left.is_empty() || right.is_empty() {
        return None;
    }

    // LHS: try full-year, then bare `M月D日`, then `本月N日` / `下个月N号` etc.
    let a = try_absolute_date_loose(left)
        .or_else(|| try_bare_month_day(left, now))
        .or_else(|| try_limit_month_day(left, now))
        .or_else(|| try_relative_year_month_day(left, now))
        .or_else(|| try_named_period(left, now))
        .or_else(|| try_relative_day(left, now))
        .or_else(|| try_fixed_holiday(left, now))
        .or_else(|| try_limit_year_festival(left, now))?;
    // For `right`, first try as a full absolute date (loose); then prefer
    // `M月D日` (inheriting left's year) BEFORE "day-only" — otherwise
    // `2017年8月11日至8月22日`'s RHS "8月22日" would mis-parse as day=8 +
    // leftover `月22日`.
    let b = if let Some(t) = try_absolute_date_loose(right) {
        t
    } else if let Some(t) = try_bare_month_day_with_year(right, a.start.year(), now) {
        t
    } else if let Some((day, rest)) = parse_day_token(right) {
        let rest = rest.trim();
        let (s, e) = date_range(a.start.year(), Some(a.start.month()), Some(day))?;
        if rest.is_empty() {
            TimeInfo {
                time_type: "time_point",
                start: s,
                end: e,
                definition: "accurate",
                ..Default::default()
            }
        } else if let Some(t) = apply_optional_clock(s, e, rest) {
            // `12日16时` — day+clock inherits year+month from LHS.
            t
        } else if let Some(month_only) = parse_bare_month(right) {
            let (start, end) = date_range(a.start.year(), Some(month_only), None)?;
            TimeInfo {
                time_type: "time_point",
                start,
                end,
                definition: "accurate",
                ..Default::default()
            }
        } else {
            return None;
        }
    } else if let Some(month_only) = parse_bare_month(right) {
        let (start, end) = date_range(a.start.year(), Some(month_only), None)?;
        TimeInfo {
            time_type: "time_point",
            start,
            end,
            definition: "accurate",
            ..Default::default()
        }
    } else if let Some(t) = try_bare_month_day(right, now) {
        // Right side has its own `M月D日` (possibly + clock), e.g.
        // `本月10日至12日16时` case is covered above via day_token branch,
        // but `本月10日至5月1日` needs this branch.
        t
    } else if let Some(t) = try_limit_month_day(right, now) {
        t
    } else if let Some(t) = try_relative_year_month_day(right, now) {
        t
    } else if let Some(t) = try_relative_day(right, now) {
        t
    } else if let Some(t) = try_fixed_holiday(right, now) {
        t
    } else if let Some(t) = try_limit_year_festival(right, now) {
        t
    } else {
        // Festival name inherited from LHS (e.g. `去年春节到元宵节` →
        // RHS is `元宵节` without year). Construct a pseudo-now anchored
        // at LHS's year so year-inheriting parsers pick up the right year.
        let lhs_now = NaiveDate::from_ymd_opt(a.start.year(), 1, 1)
            .and_then(|d| d.and_hms_opt(0, 0, 0))
            .unwrap_or(now);
        if let Some(t) = try_fixed_holiday(right, lhs_now) {
            t
        } else {
            return None;
        }
    };
    // Guard: right must come after or equal left.
    if b.end < a.start {
        return None;
    }
    // Silence unused-var warning for `now` in this simple version — may be
    // used later for implicit-year ranges like "3月5日到8日" (no year).
    let _ = now;

    // Span end: when the right side carried an explicit hour (e.g. "16时"),
    // Python returns the hour point (16:00:00) rather than the inclusive
    // end-of-hour (16:59:59). If the rhs is a time_point whose start has
    // a non-midnight clock, use `b.start` instead of `b.end`.
    let end = if b.time_type == "time_point" && b.start.time() != chrono::NaiveTime::MIN {
        b.start
    } else {
        b.end
    };

    Some(TimeInfo {
        time_type: "time_span",
        start: a.start,
        end,
        definition: "accurate",
        ..Default::default()
    })
}

// ───────────────────────── fuzzy time (Stage 5) ────────────────────────────

/// Fuzzy time expressions. Each entry maps an input to a `(signed seconds
/// offset from now, span half-width in seconds)`. The returned TimeInfo
/// has `definition = "blur"` and `start` / `end` straddling the estimated
/// point by `half_width`.
///
/// Values are deliberately rough — callers needing precise timestamps
/// should require the user to be more specific.
const FUZZY_TABLE: &[(&str, i64, i64)] = &[
    // (keyword, midpoint offset seconds, half-width seconds)
    ("刚才", -60, 30), // a minute ago ± 30s
    ("刚刚", -30, 15),
    ("刚", -60, 30),
    ("不久前", -3600, 1800), // ~1 hour ago ± 30 min
    ("不久之前", -3600, 1800),
    ("最近", -86400 * 3, 86400 * 2), // past 3 days, span ±2 days
    ("近日", -86400 * 2, 86400),
    ("近来", -86400 * 3, 86400 * 2),
    ("近期", -86400 * 7, 86400 * 5), // past ~week
    ("前不久", -3600 * 2, 3600),
    ("过一会儿", 300, 180), // in ~5 min
    ("过会儿", 300, 180),
    ("一会儿", 180, 120),
    ("等会儿", 600, 300),
    ("晚些时候", 3600 * 4, 3600 * 2), // later today
    ("晚一点", 3600 * 2, 3600),
    ("稍后", 1800, 900),
    ("马上", 60, 30),
    ("立刻", 10, 5),
    ("即将", 300, 180),
    ("不久", 3600, 1800), // in a while
];

fn try_fuzzy_time(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    // Longest-keyword wins (so "不久之前" beats "不久" and "刚才" beats "刚").
    let mut best: Option<(&str, i64, i64)> = None;
    for (kw, off, half) in FUZZY_TABLE {
        if text == *kw {
            match best {
                Some((prev, _, _)) if prev.chars().count() >= kw.chars().count() => {}
                _ => best = Some((*kw, *off, *half)),
            }
        }
    }
    let (_, offset, half) = best?;
    let mid = now + Duration::seconds(offset);
    let start = mid - Duration::seconds(half);
    let end = mid + Duration::seconds(half);
    Some(TimeInfo {
        time_type: "time_point",
        start,
        end,
        definition: "blur",
        ..Default::default()
    })
}

// ───────────────────────── time delta (Stage 4a) ───────────────────────────

/// Parse relative-offset expressions like `三天后` / `两周前` / `5分钟后`.
///
/// Shape:
///   * bare `后/前` → `time_point` at the exact moment `now ± delta`.
///   * `之后/之前/以后/以前` → `time_span` covering `[now, endpoint]` or
///     `[endpoint, now]` — matches Python's `normalize_hour_delta_point` /
///     `normalize_month_delta_point` which return spans for open-ended forms.
fn try_time_delta(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    // Direction + shape: 之后/以后 → future span; 之前/以前 → past span;
    // bare 后 → future point; bare 前 → past point.
    let (direction, is_span, body) = if let Some(rest) = text
        .strip_suffix("之后")
        .or_else(|| text.strip_suffix("以后"))
    {
        (1i64, true, rest)
    } else if let Some(rest) = text
        .strip_suffix("之前")
        .or_else(|| text.strip_suffix("以前"))
    {
        (-1, true, rest)
    } else if let Some(rest) = text.strip_suffix('后') {
        (1, false, rest)
    } else if let Some(rest) = text.strip_suffix('前') {
        (-1, false, rest)
    } else {
        return None;
    };
    let body = body.trim();

    // Split off the unit character from the body's tail.
    const UNITS: &[(&str, DeltaUnit)] = &[
        ("半小时", DeltaUnit::HalfHour),
        ("个季度", DeltaUnit::Quarter),
        ("季度", DeltaUnit::Quarter),
        ("个小时", DeltaUnit::Hour),
        ("个钟头", DeltaUnit::Hour),
        ("个月", DeltaUnit::Month),
        ("分钟", DeltaUnit::Minute),
        ("刻钟", DeltaUnit::Quarter15),
        ("星期", DeltaUnit::Week),
        ("小时", DeltaUnit::Hour),
        ("钟头", DeltaUnit::Hour),
        ("年", DeltaUnit::Year),
        ("月", DeltaUnit::Month),
        ("周", DeltaUnit::Week),
        ("天", DeltaUnit::Day),
        ("日", DeltaUnit::Day),
        ("时", DeltaUnit::Hour),
        ("分", DeltaUnit::Minute),
        ("秒", DeltaUnit::Second),
    ];

    let mut num_body: Option<&str> = None;
    let mut matched_unit: Option<DeltaUnit> = None;
    for (suffix, unit) in UNITS {
        if let Some(rest) = body.strip_suffix(suffix) {
            num_body = Some(rest);
            matched_unit = Some(*unit);
            break;
        }
    }
    let num_body = num_body?.trim_end_matches('个').trim();
    let unit = matched_unit?;

    // HalfHour already represents "half of an hour" — the implicit count
    // is 1. The seconds calculation in apply_delta handles the 1800-second
    // base.
    let count: f64 = if matches!(unit, DeltaUnit::HalfHour) {
        1.0
    } else {
        parse_count(num_body)?
    };

    let result = apply_delta(now, count * direction as f64, unit)?;
    // Year-unit bare `前/后` → time_span covering the target year (Python
    // parity with pattern #22 `32年前`).
    let year_span_bare = !is_span && matches!(unit, DeltaUnit::Year);
    if year_span_bare {
        let target_year = now.year() + (count as i32) * direction as i32;
        let first = NaiveDate::from_ymd_opt(target_year, 1, 1)?;
        let last = NaiveDate::from_ymd_opt(target_year, 12, 31)?;
        return Some(TimeInfo {
            time_type: "time_span",
            start: first.and_hms_opt(0, 0, 0)?,
            end: last.and_hms_opt(23, 59, 59)?,
            definition: "blur",
            ..Default::default()
        });
    }
    if is_span {
        let (start, end) = if direction > 0 {
            (now, result)
        } else {
            (result, now)
        };
        Some(TimeInfo {
            time_type: "time_span",
            start,
            end,
            definition: "blur",
            ..Default::default()
        })
    } else {
        Some(TimeInfo {
            time_type: "time_point",
            start: result,
            end: result,
            definition: "accurate",
            ..Default::default()
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum DeltaUnit {
    Second,
    Minute,
    Quarter15, // 15-minute unit (`刻钟`).
    HalfHour,
    Hour,
    Day,
    Week,
    Month,
    Quarter,
    Year,
}

fn parse_count(body: &str) -> Option<f64> {
    let body = body.trim();
    if body.is_empty() {
        return None;
    }
    // Try pure ASCII digits (incl. decimal) first.
    if let Ok(n) = body.parse::<f64>() {
        return Some(n);
    }
    // Try Chinese numeric expression via char2num — supports "三" → 3 and
    // "两" → 2 plus the usual big-unit stacks.
    crate::gadget::money_num2char::char2num(body).ok()
}

fn apply_delta(now: NaiveDateTime, count: f64, unit: DeltaUnit) -> Option<NaiveDateTime> {
    let seconds = match unit {
        DeltaUnit::Second => count,
        DeltaUnit::Minute => count * 60.0,
        DeltaUnit::Quarter15 => count * 900.0,
        DeltaUnit::HalfHour => count * 1800.0,
        DeltaUnit::Hour => count * 3600.0,
        DeltaUnit::Day => count * 86_400.0,
        DeltaUnit::Week => count * 604_800.0,
        DeltaUnit::Month => {
            // Month arithmetic preserves day-of-month where possible.
            let months = count as i32;
            if (count - months as f64).abs() > 1e-9 {
                // Fractional months fall back to approximate 30-day math.
                count * 86_400.0 * 30.0
            } else {
                return add_months(now, months);
            }
        }
        DeltaUnit::Year => {
            let years = count as i32;
            if (count - years as f64).abs() > 1e-9 {
                count * 86_400.0 * 365.25
            } else {
                return add_months(now, years * 12);
            }
        }
        DeltaUnit::Quarter => {
            let quarters = count as i32;
            if (count - quarters as f64).abs() > 1e-9 {
                count * 86_400.0 * 90.0
            } else {
                return add_months(now, quarters * 3);
            }
        }
    };
    let dur = chrono::Duration::seconds(seconds as i64);
    Some(now + dur)
}

fn add_months(base: NaiveDateTime, n: i32) -> Option<NaiveDateTime> {
    let mut y = base.year();
    let mut m = base.month() as i32 + n;
    while m <= 0 {
        m += 12;
        y -= 1;
    }
    while m > 12 {
        m -= 12;
        y += 1;
    }
    let d = base.day();
    // Clamp day to month-end if target month has fewer days.
    let mut final_day = d;
    let mut date = NaiveDate::from_ymd_opt(y, m as u32, final_day);
    while date.is_none() && final_day > 28 {
        final_day -= 1;
        date = NaiveDate::from_ymd_opt(y, m as u32, final_day);
    }
    let date = date?;
    date.and_time(base.time()).into()
}

// ───────────────────────── named period (Stage 4b) ──────────────────────────

/// Parse 本/上/下/今/去 + 周/月/年/季度.
fn try_named_period(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    // Intentionally omit `明/后/前` as single-char prefixes — those clash
    // with 明天/后天/前天 which have a dedicated path in `try_relative_day`.
    // `去` is safe (去年 has no relative-day collision).
    const OFFSETS: &[(&str, i32)] = &[
        ("本", 0),
        ("这", 0),
        ("今", 0),
        ("当", 0),
        ("上上一个", -2),
        ("上上个", -2),
        ("上上一", -2),
        ("上上", -2),
        ("上一个", -1),
        ("上个", -1),
        ("上一", -1),
        ("上", -1),
        ("去", -1),
        ("下下一个", 2),
        ("下下个", 2),
        ("下下一", 2),
        ("下下", 2),
        ("下一个", 1),
        ("下个", 1),
        ("下一", 1),
        ("下", 1),
    ];
    let mut matched: Option<(i32, &str)> = None;
    for (prefix, off) in OFFSETS {
        if let Some(rest) = text.strip_prefix(*prefix) {
            // Longest prefix wins: "上个月" should pick "上个" not "上".
            match matched {
                Some((_, r)) if r.len() < rest.len() => {}
                _ => matched = Some((*off, rest)),
            }
        }
    }
    let (offset, rest) = matched?;
    // Track which prefix matched so we can apply Python's time_point vs
    // time_span classification by prefix shape.
    let matched_prefix_emphasizes_point = text.starts_with("当")
        || text.starts_with("上一个")
        || text.starts_with("下一个")
        || text.starts_with("上一")
        || text.starts_with("下一");
    // Tolerate a leading `个` — covers `上一个月` matching via `上一` with
    // rest `个月`. Also strip surrounding whitespace. Also tolerate a
    // leading `一` so `这一年` matches via `这` prefix + rest `一年`.
    let rest = rest
        .trim_start_matches('个')
        .trim_start_matches('一')
        .trim();

    // Deliberately exclude 天/日 from this path — relative-day handles
    // 今天/明天/昨天 etc. This parser only covers calendar periods
    // (week/month/quarter/year). The tail after the period unit must also
    // be empty or trivial — we don't want to swallow "本月3日" as
    // "this-month" when the user meant "the 3rd of this month".
    let (period, is_point): (NamedPeriod, bool) =
        if rest == "周" || rest == "星期" || rest == "礼拜" {
            // Python returns time_point for all named weeks.
            (NamedPeriod::Week, true)
        } else if rest == "月" {
            // Python returns time_point for `当月` and `上一个月` / `下一个月`;
            // time_span for generic `本月` / `这个月`.
            (NamedPeriod::Month, matched_prefix_emphasizes_point)
        } else if rest == "年" {
            (NamedPeriod::Year, false)
        } else if rest == "季度" {
            (NamedPeriod::Quarter, false)
        } else {
            return None;
        };

    let (start, end) = period_range(period, now.date(), offset)?;
    Some(TimeInfo {
        time_type: if is_point { "time_point" } else { "time_span" },
        start,
        end,
        definition: "accurate",
        ..Default::default()
    })
}

#[derive(Debug, Clone, Copy)]
enum NamedPeriod {
    Week,
    Month,
    Quarter,
    Year,
}

fn period_range(
    period: NamedPeriod,
    today: NaiveDate,
    offset: i32,
) -> Option<(NaiveDateTime, NaiveDateTime)> {
    match period {
        NamedPeriod::Week => {
            // Monday of current week:
            let dow = today.weekday().num_days_from_monday() as i64;
            let this_mon = today - Duration::days(dow);
            let start_mon = this_mon + Duration::days(offset as i64 * 7);
            let end_sun = start_mon + Duration::days(6);
            Some((
                start_mon.and_hms_opt(0, 0, 0)?,
                end_sun.and_hms_opt(23, 59, 59)?,
            ))
        }
        NamedPeriod::Month => {
            let mut y = today.year();
            let mut m = today.month() as i32 + offset;
            while m <= 0 {
                m += 12;
                y -= 1;
            }
            while m > 12 {
                m -= 12;
                y += 1;
            }
            let first = NaiveDate::from_ymd_opt(y, m as u32, 1)?;
            let next_first = if m == 12 {
                NaiveDate::from_ymd_opt(y + 1, 1, 1)
            } else {
                NaiveDate::from_ymd_opt(y, (m + 1) as u32, 1)
            }?;
            let last = next_first.pred_opt()?;
            Some((first.and_hms_opt(0, 0, 0)?, last.and_hms_opt(23, 59, 59)?))
        }
        NamedPeriod::Quarter => {
            let cur_q = ((today.month() - 1) / 3) as i32; // 0..3
            let mut q = cur_q + offset;
            let mut y = today.year();
            while q < 0 {
                q += 4;
                y -= 1;
            }
            while q > 3 {
                q -= 4;
                y += 1;
            }
            let start_month = (q * 3 + 1) as u32;
            let first = NaiveDate::from_ymd_opt(y, start_month, 1)?;
            let end_month = start_month + 2;
            let (end_year, end_month_norm) = if end_month > 12 {
                (y + 1, end_month - 12)
            } else {
                (y, end_month)
            };
            let next = if end_month_norm == 12 {
                NaiveDate::from_ymd_opt(end_year + 1, 1, 1)
            } else {
                NaiveDate::from_ymd_opt(end_year, end_month_norm + 1, 1)
            }?;
            let last = next.pred_opt()?;
            Some((first.and_hms_opt(0, 0, 0)?, last.and_hms_opt(23, 59, 59)?))
        }
        NamedPeriod::Year => {
            let y = today.year() + offset;
            let first = NaiveDate::from_ymd_opt(y, 1, 1)?;
            let last = NaiveDate::from_ymd_opt(y, 12, 31)?;
            Some((first.and_hms_opt(0, 0, 0)?, last.and_hms_opt(23, 59, 59)?))
        }
    }
}

// ───────────────────────── clock range (Stage 3a) ──────────────────────────

/// Try to parse a clock-range expression. Accepts:
///   "8点到12点"            — today's 08:00..12:00
///   "上午8点到下午3点"     — disambiguates via the period qualifiers
///   "明天下午3点到5点"     — relative-day prefix + clock range
///   "2024年3月5日上午8点到12点"
fn try_clock_range(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    // Pull off an optional date prefix: if the text starts with a
    // recognizable date or relative-day, keep that as the base day.
    let (base_day, rest) = extract_date_prefix(text, now);
    let rest = rest.trim();

    // Need one of the range separators with a clock on each side.
    for sep in ["——", "--", "到", "至", "—", "-", "~", "～"] {
        if let Some(idx) = rest.find(sep) {
            let left = rest[..idx].trim();
            let right = rest[idx + sep.len()..].trim();
            if left.is_empty() || right.is_empty() {
                continue;
            }
            // Both sides must parse as clock. We also require the right
            // side to NOT contain a 日/号 (that would be a date range), and
            // both sides must have a clock marker (点/时/:) to prevent
            // greedy digit-matching on year-month strings like "2024年3月".
            if right.contains('日')
                || right.contains('号')
                || right.contains('年')
                || left.contains('年')
            {
                continue;
            }
            let has_clock_marker =
                |s: &str| s.contains('点') || s.contains('时') || s.contains(':');
            if !has_clock_marker(left) && !has_clock_marker(right) {
                continue;
            }
            let left_clock = parse_clock(left)?;
            // If the right side doesn't carry its own period qualifier
            // (上午/下午/…) but the left side does, inherit the left's
            // qualifier. Otherwise parse the right side as-is.
            let (right_period, _) = split_period(right);
            let (left_period, _) = split_period(left);
            let right_clock = match (right_period, left_period) {
                (None, Some(lp)) => parse_clock(&format!("{}{}", lp, right))?,
                _ => parse_clock(right)?,
            };

            let start = base_day.and_time(left_clock);
            let end = base_day.and_time(right_clock);
            if end < start {
                return None;
            }
            return Some(TimeInfo {
                time_type: "time_span",
                start,
                end,
                definition: "accurate",
                ..Default::default()
            });
        }
    }
    None
}

/// Return (base_day, remainder) where base_day is today if no explicit
/// date prefix is found. The remainder is the text with the date prefix
/// stripped off.
fn extract_date_prefix(text: &str, now: NaiveDateTime) -> (NaiveDate, &str) {
    // Try absolute date prefix.
    if let Some(caps) = YMD_CN.captures(text) {
        if let (Some(y), Some(m), Some(d)) = (caps.get(1), caps.get(2), caps.get(3)) {
            if let (Some(year), Some(month), Some(day)) = (
                parse_year(y.as_str()),
                m.as_str().parse::<u32>().ok(),
                d.as_str().parse::<u32>().ok(),
            ) {
                if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
                    let rest = caps.get(4).map(|x| x.as_str()).unwrap_or("");
                    return (date, rest);
                }
            }
        }
    }
    // Try relative-day prefix.
    const DAYS: &[(&str, i64)] = &[
        ("大前天", -3),
        ("大后天", 3),
        ("前天", -2),
        ("后天", 2),
        ("昨天", -1),
        ("昨日", -1),
        ("明天", 1),
        ("明日", 1),
        ("今天", 0),
        ("今日", 0),
    ];
    for (kw, offset) in DAYS {
        if let Some(rest) = text.strip_prefix(*kw) {
            return (now.date() + Duration::days(*offset), rest);
        }
    }
    (now.date(), text)
}

/// Split off a leading period qualifier (上午/下午/…) from a clock string.
/// Returns `(Some(period), remainder)` or `(None, whole)`.
fn split_period(s: &str) -> (Option<&str>, &str) {
    const PERIODS: &[&str] = &[
        "凌晨", "早上", "早晨", "早", "上午", "中午", "下午", "午后", "晚上", "傍晚", "夜里",
        "夜间",
    ];
    for p in PERIODS {
        if let Some(rest) = s.strip_prefix(*p) {
            return (Some(p), rest);
        }
    }
    (None, s)
}

// ───────────────────────── recurring (Stage 3b) ─────────────────────────────

/// Parse "每X" expressions. Returns a `time_period` whose `delta` encodes
/// the recurrence cadence (`{day: 7}` for weekly, `{day: 1}` for daily,
/// etc.) and `point` pins each anchor within one cycle.
fn try_recurring(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let body = text.strip_prefix('每')?;
    let body = body.trim_start();

    // Weekday: 周一/…/周日 or 星期一/…/星期天.
    if let Some((weekday, rest)) = match_weekday(body) {
        let next = next_weekday(now.date(), weekday);
        return build_period(next, rest.trim(), cadence_every_days(7.0));
    }

    // Day-of-month: 每月N日/号.
    if let Some(rest) = body.strip_prefix('月') {
        let rest = rest.trim_start();
        let (day, tail) = take_leading_day(rest)?;
        let next = next_day_of_month(now.date(), day)?;
        return build_period(next, tail.trim(), cadence_every_months(1.0));
    }

    // Day of week shorthand "每天" or "每日".
    if let Some(rest) = body.strip_prefix('天').or_else(|| body.strip_prefix('日')) {
        let next = now.date() + Duration::days(1);
        return build_period(next, rest.trim(), cadence_every_days(1.0));
    }

    None
}

fn cadence_every_days(n: f64) -> TimeDelta {
    let mut d = TimeDelta::default();
    d.day = Some(DeltaValue::Single(n));
    d
}

fn cadence_every_months(n: f64) -> TimeDelta {
    let mut d = TimeDelta::default();
    d.month = Some(DeltaValue::Single(n));
    d
}

/// Build a `time_period` `TimeInfo` for the given anchor date and a trailing
/// clock / clock-range / empty tail.
fn build_period(next: NaiveDate, tail: &str, delta: TimeDelta) -> Option<TimeInfo> {
    let (start, end, point_string) = if tail.is_empty() {
        (
            next.and_hms_opt(0, 0, 0)?,
            next.and_hms_opt(23, 59, 59)?,
            String::new(),
        )
    } else if let Some((lc, rc)) = parse_clock_range(tail) {
        (next.and_time(lc), next.and_time(rc), tail.to_string())
    } else {
        let c = parse_clock(tail)?;
        let dt = next.and_time(c);
        (dt, dt, tail.to_string())
    };

    Some(TimeInfo {
        time_type: "time_period",
        start,
        end,
        definition: "accurate",
        delta: Some(delta.clone()),
        period: Some(TimePeriodInfo {
            delta,
            point_time: vec![(start, end)],
            point_string,
        }),
    })
}

/// Match a weekday prefix and return its 0-6 index (0 = Monday, 6 = Sunday)
/// plus the remainder.
fn match_weekday(s: &str) -> Option<(u32, &str)> {
    const TABLE: &[(&str, u32)] = &[
        ("星期一", 0),
        ("星期二", 1),
        ("星期三", 2),
        ("星期四", 3),
        ("星期五", 4),
        ("星期六", 5),
        ("星期日", 6),
        ("星期天", 6),
        ("周一", 0),
        ("周二", 1),
        ("周三", 2),
        ("周四", 3),
        ("周五", 4),
        ("周六", 5),
        ("周日", 6),
        ("周天", 6),
    ];
    for (pat, idx) in TABLE {
        if let Some(rest) = s.strip_prefix(*pat) {
            return Some((*idx, rest));
        }
    }
    None
}

fn next_weekday(from: NaiveDate, target: u32) -> NaiveDate {
    // Mon=0..Sun=6 in our table; chrono's Weekday::num_days_from_monday()
    // matches.
    let cur = from.weekday().num_days_from_monday();
    let mut delta = (target + 7 - cur) % 7;
    if delta == 0 {
        delta = 7;
    }
    from + Duration::days(delta as i64)
}

fn take_leading_day(s: &str) -> Option<(u32, &str)> {
    // Parse leading ASCII digits for day-of-month (1..31), then optional 日/号.
    let bytes = s.as_bytes();
    let mut end = 0usize;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end == 0 {
        return None;
    }
    let day: u32 = s[..end].parse().ok()?;
    if !(1..=31).contains(&day) {
        return None;
    }
    let mut rest = &s[end..];
    rest = rest
        .strip_prefix('日')
        .or_else(|| rest.strip_prefix('号'))
        .unwrap_or(rest);
    Some((day, rest))
}

fn next_day_of_month(from: NaiveDate, day: u32) -> Option<NaiveDate> {
    // Try this month first; if invalid (e.g. 31st in Feb) or past, jump forward.
    let try_in = |year: i32, month: u32| NaiveDate::from_ymd_opt(year, month, day);
    if let Some(d) = try_in(from.year(), from.month()) {
        if d > from {
            return Some(d);
        }
    }
    // Walk forward until we find a valid month containing `day`.
    let mut year = from.year();
    let mut month = from.month();
    for _ in 0..24 {
        month += 1;
        if month > 12 {
            month = 1;
            year += 1;
        }
        if let Some(d) = try_in(year, month) {
            return Some(d);
        }
    }
    None
}

fn try_clock_only(text: &str, today: NaiveDate) -> Option<TimeInfo> {
    // If `text` is *only* a clock expression (no leading year/day), parse it
    // as today's clock. We guard with a quick reject for strings that begin
    // with purely Hanzi that isn't a period marker.
    let clock = parse_clock(text)?;
    // Require the match to cover the whole length, otherwise we risk
    // misreading `2024` as hour 20. CLOCK is anchored at start; enforce
    // end by re-matching — but first strip our Chinese fractional-hour
    // suffixes so their tail doesn't trip the "extra trailing text" check.
    let normalized = text
        .strip_suffix("点半")
        .or_else(|| text.strip_suffix("时半"))
        .or_else(|| text.strip_suffix("点一刻"))
        .or_else(|| text.strip_suffix("点二刻"))
        .or_else(|| text.strip_suffix("点两刻"))
        .or_else(|| text.strip_suffix("点三刻"))
        .map(|s| format!("{}点", s))
        .unwrap_or_else(|| text.to_string());
    let caps = CLOCK.captures(&normalized)?;
    if caps.get(0)?.as_str().chars().count() < normalized.chars().count() - 1
        && !normalized[caps.get(0)?.end()..].trim().is_empty()
    {
        return None;
    }
    let dt = today.and_time(clock);
    Some(TimeInfo {
        time_type: "time_point",
        start: dt,
        end: dt,
        definition: "accurate",
        ..Default::default()
    })
}

// ───────────────────────── Chinese numeral helpers ─────────────────────────

/// Parse a small Chinese numeral 1..=99 used in counts ("两个", "三十五").
/// Does not accept the zero/〇 form — counts are always positive.
fn cn_int(s: &str) -> Option<u32> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let digit = |c: char| -> Option<u32> {
        match c {
            '一' => Some(1),
            '二' | '两' => Some(2),
            '三' => Some(3),
            '四' => Some(4),
            '五' => Some(5),
            '六' => Some(6),
            '七' => Some(7),
            '八' => Some(8),
            '九' => Some(9),
            _ => None,
        }
    };
    let chars: Vec<char> = s.chars().collect();
    match chars.as_slice() {
        [a, '十', b] => {
            let t = digit(*a)?;
            let u = digit(*b)?;
            Some(t * 10 + u)
        }
        [a, '十'] => {
            let t = digit(*a)?;
            Some(t * 10)
        }
        ['十', b] => {
            let u = digit(*b)?;
            Some(10 + u)
        }
        ['十'] => Some(10),
        // `廿`/`念` (both = 20), `卅`/`丗` (30), optionally + ones digit.
        ['廿' | '念', b] => {
            let u = digit(*b)?;
            Some(20 + u)
        }
        ['廿' | '念'] => Some(20),
        ['卅' | '丗', b] => {
            let u = digit(*b)?;
            Some(30 + u)
        }
        ['卅' | '丗'] => Some(30),
        [a] => digit(*a),
        _ => None,
    }
}

/// Parse a leading Chinese or Arabic day-of-month (1..=31), consuming an
/// optional 日 / 号 suffix. Returns `(day, rest)`.
fn parse_day_token(s: &str) -> Option<(u32, &str)> {
    let s = s.trim_start();
    // Arabic first.
    let bytes = s.as_bytes();
    let mut end = 0usize;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end > 0 {
        let day: u32 = s[..end].parse().ok()?;
        if !(1..=31).contains(&day) {
            return None;
        }
        let rest = &s[end..];
        let rest = rest
            .strip_prefix('日')
            .or_else(|| rest.strip_prefix('号'))
            .unwrap_or(rest);
        return Some((day, rest));
    }
    // Chinese: consume up to 3 chars matching a known pattern.
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    for take in [3usize, 2, 1] {
        if chars.len() < take {
            continue;
        }
        let end_byte = if chars.len() == take {
            s.len()
        } else {
            chars[take].0
        };
        let chunk = &s[..end_byte];
        if let Some(n) = cn_int(chunk) {
            if (1..=31).contains(&n) {
                let rest = &s[end_byte..];
                let rest = rest
                    .strip_prefix('日')
                    .or_else(|| rest.strip_prefix('号'))
                    .unwrap_or(rest);
                return Some((n, rest));
            }
        }
    }
    None
}

// ───────────────────────── loose absolute date ────────────────────────────

/// Like `try_absolute_date` but additionally accepts Chinese-numeral days
/// in the "YYYY年MM月<CN_DAY>[日|号]" form.
fn try_absolute_date_loose(text: &str) -> Option<TimeInfo> {
    if let Some(t) = try_absolute_date(text) {
        return Some(t);
    }
    // `YYYY年MM月<day?>` with Chinese numerals permitted for month/day.
    static RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"^(\d{2,4})\s*年\s*(\d{1,2}|[零〇一二三四五六七八九十]{1,3})\s*月\s*(.*)$")
            .unwrap()
    });
    if let Some(caps) = RE.captures(text) {
        let year = parse_year(caps.get(1)?.as_str())?;
        let m_str = caps.get(2)?.as_str();
        let month: u32 = m_str.parse::<u32>().ok().or_else(|| cn_int(m_str))?;
        if !(1..=12).contains(&month) {
            return None;
        }
        let rest = caps.get(3)?.as_str().trim();
        if rest.is_empty() {
            // Month-only → whole month as time_span.
            let (start, end) = date_range(year, Some(month), None)?;
            return Some(TimeInfo {
                time_type: "time_span",
                start,
                end,
                definition: "accurate",
                ..Default::default()
            });
        }
        if let Some((day, tail)) = parse_day_token(rest) {
            let (start, end) = date_range(year, Some(month), Some(day))?;
            return apply_optional_clock(start, end, tail.trim());
        }
        // Rest is clock-only after the month (`2019年5月15:20`)? Skip.
    }
    None
}

// ───────────────────────── year + span month / quarter ─────────────────────

/// Pattern #9 — "YYYY年(前|后|头|最后|开头|最初|最末)N个?(月|季度)".
/// Returns a `time_span` covering the first-N or last-N months of the year.
fn try_year_span_month_or_quarter(text: &str, _now: NaiveDateTime) -> Option<TimeInfo> {
    static RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"^(\d{2,4})\s*年\s*(前|后|头|最后|开头|最初|最末)\s*(\d+|[一二两三四五六七八九十]+)\s*个?\s*(季度|月)$",
        )
        .unwrap()
    });
    let caps = RE.captures(text)?;
    let year = parse_year(caps.get(1)?.as_str())?;
    let direction = caps.get(2)?.as_str();
    let n_str = caps.get(3)?.as_str();
    let n: u32 = n_str.parse::<u32>().ok().or_else(|| cn_int(n_str))?;
    let unit = caps.get(4)?.as_str();

    let months = if unit == "季度" { n * 3 } else { n };
    if months == 0 || months > 12 {
        return None;
    }

    let is_forward = matches!(direction, "前" | "头" | "开头" | "最初");
    let (first_month, last_month) = if is_forward {
        (1u32, months)
    } else {
        (13 - months, 12)
    };
    let first = NaiveDate::from_ymd_opt(year, first_month, 1)?;
    let next_last = if last_month == 12 {
        NaiveDate::from_ymd_opt(year + 1, 1, 1)?
    } else {
        NaiveDate::from_ymd_opt(year, last_month + 1, 1)?
    };
    let last = next_last.pred_opt()?;
    Some(TimeInfo {
        time_type: "time_span",
        start: first.and_hms_opt(0, 0, 0)?,
        end: last.and_hms_opt(23, 59, 59)?,
        definition: "accurate",
        ..Default::default()
    })
}

// ───────────────────────── time_delta range ────────────────────────────────

/// Pattern #90 — "N[~至到]M<unit>" produces a `time_delta` whose unit is
/// `Range(N, M)`. Examples: `30~90日`, `2到5年`, `10-15分钟`.
fn try_delta_range(text: &str, _now: NaiveDateTime) -> Option<TimeInfo> {
    static RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"^(\d+(?:\.\d+)?)\s*[~～\-—至到]\s*(\d+(?:\.\d+)?)\s*(个小时|个月|分钟|小时|星期|工作日|秒|日|天|年|月|周)$",
        )
        .unwrap()
    });
    let caps = RE.captures(text)?;
    let lo: f64 = caps.get(1)?.as_str().parse().ok()?;
    let hi: f64 = caps.get(2)?.as_str().parse().ok()?;
    if hi < lo {
        return None;
    }
    let unit = caps.get(3)?.as_str();

    let mut d = TimeDelta::default();
    let v = DeltaValue::Range(lo, hi);
    match unit {
        "日" | "天" => d.day = Some(v),
        "年" => d.year = Some(v),
        "个小时" | "小时" => d.hour = Some(v),
        "分钟" => d.minute = Some(v),
        "秒" => d.second = Some(v),
        "个月" | "月" => d.month = Some(v),
        "周" | "星期" => d.day = Some(DeltaValue::Range(lo * 7.0, hi * 7.0)),
        "工作日" => d.workday = Some(v),
        _ => return None,
    }

    Some(TimeInfo {
        time_type: "time_delta",
        definition: "blur",
        delta: Some(d),
        ..Default::default()
    })
}

// ───────────────────────── clock range helper ──────────────────────────────

// ───────────────────────── Round 17 — easy-win parsers ────────────────────

/// Pattern #2 — 8-digit date `20210901`. Strict — exactly 8 ASCII digits.
fn try_eight_digit_ymd(text: &str) -> Option<TimeInfo> {
    if text.len() != 8 || !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let year: i32 = text[..4].parse().ok()?;
    let month: u32 = text[4..6].parse().ok()?;
    let day: u32 = text[6..8].parse().ok()?;
    if !(1900..=2100).contains(&year) || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let (start, end) = date_range(year, Some(month), Some(day))?;
    Some(TimeInfo {
        time_type: "time_point",
        start,
        end,
        definition: "accurate",
        ..Default::default()
    })
}

/// Pattern #11 — `YYYY年第N季度` / `YYYY年N季度` / `YYYY年Q<n>` /
/// `YYYY年首季度` / `一季度` (bare, inherits year from `now`) /
/// Chinese-year + quarter.
fn try_year_solar_season(text: &str) -> Option<TimeInfo> {
    // Case 1: Arabic year + quarter (+ optional 首).
    static RE_Y: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"^(\d{2,4})\s*年\s*(?:第)?\s*(首|[1-4一二三四])\s*季度$").unwrap()
    });
    // Case 2: Chinese year + quarter.
    static RE_CN_Y: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"^([零〇一二三四五六七八九十两]+)\s*年\s*(?:第)?\s*(首|[1-4一二三四])\s*季度$")
            .unwrap()
    });
    // Case 3: Bare quarter (no year).
    static RE_BARE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"^(?:第)?\s*(首|[1-4一二三四])\s*季度$").unwrap());
    // Case 4: Q<n>季度 form.
    static RE_Q: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(?:[Qq])([1-4])\s*季度$").unwrap());

    let (year, q) = if let Some(caps) = RE_Y.captures(text) {
        let y = parse_year(caps.get(1)?.as_str())?;
        let n = parse_quarter_token(caps.get(2)?.as_str())?;
        (y, n)
    } else if let Some(caps) = RE_CN_Y.captures(text) {
        let y = parse_chinese_year_digits(caps.get(1)?.as_str())?;
        let n = parse_quarter_token(caps.get(2)?.as_str())?;
        (y, n)
    } else if let Some(caps) = RE_Q.captures(text) {
        // Q1季度 inherits year from now — handled by try_year_solar_season
        // being dispatched only when now is available? Not yet; for now,
        // fall through to None and let the bare path in the parse pipeline
        // handle it. Since this branch reaches us without `now`, we cannot
        // resolve; return None here and let try_bare_quarter (dispatched
        // separately) handle it.
        let _ = caps;
        return None;
    } else if RE_BARE.is_match(text) {
        // Same as above — cannot resolve without `now`.
        return None;
    } else {
        return None;
    };

    if !(1..=4).contains(&q) {
        return None;
    }
    let start_month = (q - 1) * 3 + 1;
    let end_month = start_month + 2;
    let first = NaiveDate::from_ymd_opt(year, start_month, 1)?;
    let next = if end_month == 12 {
        NaiveDate::from_ymd_opt(year + 1, 1, 1)?
    } else {
        NaiveDate::from_ymd_opt(year, end_month + 1, 1)?
    };
    let last = next.pred_opt()?;
    Some(TimeInfo {
        time_type: "time_span",
        start: first.and_hms_opt(0, 0, 0)?,
        end: last.and_hms_opt(23, 59, 59)?,
        definition: "accurate",
        ..Default::default()
    })
}

/// Parse `首` / `一二三四` / `1..4` as a quarter number (1..=4).
/// Python parity: Python accepts 首 as "first" (= 1).
fn parse_quarter_token(s: &str) -> Option<u32> {
    if s == "首" {
        return Some(1);
    }
    s.parse::<u32>().ok().or_else(|| cn_int(s))
}

/// Bare quarter without year — `一季度`, `首季度`, `Q1季度` inherits
/// year from `now`. Split from try_year_solar_season because it needs
/// `now` whereas that function is called both with and without it.
fn try_bare_quarter(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    static RE_BARE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"^(?:第)?\s*(首|[1-4一二三四])\s*季度$").unwrap());
    static RE_Q: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[Qq]([1-4])\s*季度$").unwrap());

    let q_str = if let Some(caps) = RE_BARE.captures(text) {
        caps.get(1)?.as_str().to_string()
    } else if let Some(caps) = RE_Q.captures(text) {
        caps.get(1)?.as_str().to_string()
    } else {
        return None;
    };
    let q = parse_quarter_token(&q_str)?;
    if !(1..=4).contains(&q) {
        return None;
    }
    let year = now.year();
    let start_month = (q - 1) * 3 + 1;
    let end_month = start_month + 2;
    let first = NaiveDate::from_ymd_opt(year, start_month, 1)?;
    let next = if end_month == 12 {
        NaiveDate::from_ymd_opt(year + 1, 1, 1)?
    } else {
        NaiveDate::from_ymd_opt(year, end_month + 1, 1)?
    };
    let last = next.pred_opt()?;
    Some(TimeInfo {
        time_type: "time_span",
        start: first.and_hms_opt(0, 0, 0)?,
        end: last.and_hms_opt(23, 59, 59)?,
        definition: "accurate",
        ..Default::default()
    })
}

/// Delta-range + 之后/之前/以前/以后 suffix. Python's convention:
///   `几十年之后` → [now + 20y, now + 100y]
///   `二十几年前` → [now - 30y, now - 20y]
///   `1000多年之后` → [now + 1000y, now + 1100y] (skipped via inf sentinel)
fn try_delta_range_suffix(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let (is_after, body) = if let Some(b) = text
        .strip_suffix("之后")
        .or_else(|| text.strip_suffix("以后"))
        .or_else(|| text.strip_suffix('后'))
    {
        (true, b)
    } else if let Some(b) = text
        .strip_suffix("之前")
        .or_else(|| text.strip_suffix("以前"))
        .or_else(|| text.strip_suffix('前'))
    {
        (false, b)
    } else {
        return None;
    };
    let body = body.trim();
    // Parse body as pure delta and require it to have a range (not point).
    let inner = try_pure_delta(body)?;
    if inner.time_type != "time_delta" {
        return None;
    }
    let delta = inner.delta?;
    // Only handle year/month here — day/hour spans are tiny and less
    // commonly requested in this shape.
    let (lo_years, hi_years) = match delta.year {
        Some(DeltaValue::Range(lo, hi)) => (lo, hi),
        Some(DeltaValue::Single(n)) => (n, n),
        _ => return None,
    };
    if lo_years.is_sign_negative() || hi_years.is_sign_negative() {
        return None;
    }
    let y1 = if is_after {
        now.year() + lo_years as i32
    } else {
        now.year() - hi_years as i32
    };
    let y2 = if is_after {
        now.year() + hi_years as i32
    } else {
        now.year() - lo_years as i32
    };
    let first = NaiveDate::from_ymd_opt(y1, 1, 1)?;
    let last = NaiveDate::from_ymd_opt(y2, 12, 31)?;
    Some(TimeInfo {
        time_type: "time_span",
        start: first.and_hms_opt(0, 0, 0)?,
        end: last.and_hms_opt(23, 59, 59)?,
        definition: "blur",
        ..Default::default()
    })
}

/// Concrete time + 左右 / 前后 / 附近. Strip the modifier, reparse,
/// change definition to `blur`. Python keeps the same concrete result.
fn try_approx_modifier(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    for suf in ["左右", "前后", "附近"] {
        if let Some(body) = text.strip_suffix(suf) {
            let body = body.trim();
            if body.is_empty() {
                continue;
            }
            let inner = parse_time_with_ref(body, now)?;
            return Some(TimeInfo {
                definition: "blur",
                ..inner
            });
        }
    }
    None
}

/// `从X起[至今|到现在|到今天]` / `X至今` / `X到现在`. Returns a time_span
/// from X's start to `now`.
fn try_span_to_now(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    // Strip leading 从/自, strailing 起.
    let mut body = text.trim();
    body = body.strip_prefix('从').unwrap_or(body);
    body = body.strip_prefix('自').unwrap_or(body);

    // Accept one of the `...到现在` / `...至今` / `...到今天` endings.
    // Order longest-first for unambiguous match.
    const TRAILS: &[&str] = &["至现在", "至今天", "到现在", "到今天", "至今"];
    let mut matched: Option<&str> = None;
    for t in TRAILS {
        if let Some(left) = body.strip_suffix(*t) {
            matched = Some(left);
            break;
        }
    }
    // Also accept `从X起` alone (without the `至今` tail) — Python treats
    // it as "from X to now".
    let left = if let Some(left) = matched {
        left
    } else if let Some(left) = body.strip_suffix('起') {
        left
    } else {
        return None;
    };
    let left = left.trim();
    if left.is_empty() {
        return None;
    }
    // Parse `left` as a concrete time. Must produce point/span.
    let inner = parse_time_with_ref(left, now)?;
    if !(inner.time_type == "time_point" || inner.time_type == "time_span") {
        return None;
    }
    // Span from inner's start to `now` — guard against now-before-inner.
    if now < inner.start {
        return None;
    }
    Some(TimeInfo {
        time_type: "time_span",
        start: inner.start,
        end: now,
        definition: "accurate",
        ..Default::default()
    })
}

/// `同月D号[clock]` / `同年M月D号`. Inherits year/month from `now`.
fn try_same_month_or_year(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    if let Some(rest) = text.strip_prefix("同月") {
        let (day, tail) = parse_day_token(rest)?;
        let year = now.year();
        let month = now.month();
        let (start, end) = date_range(year, Some(month), Some(day))?;
        return apply_optional_clock(start, end, tail.trim());
    }
    if let Some(rest) = text.strip_prefix("同年") {
        let (month, rest) = parse_month_token(rest)?;
        let year = now.year();
        if rest.is_empty() {
            let (start, end) = date_range(year, Some(month), None)?;
            return Some(TimeInfo {
                time_type: "time_span",
                start,
                end,
                definition: "accurate",
                ..Default::default()
            });
        }
        let (day, tail) = parse_day_token(rest)?;
        let (start, end) = date_range(year, Some(month), Some(day))?;
        return apply_optional_clock(start, end, tail.trim());
    }
    None
}

/// `<year>?<M月>上旬/中旬/下旬/月底`. Ten-day third-of-month granularity.
///   上旬: days 1-10, 中旬: 11-20, 下旬: 21-EOM.
///   月底: last 5 days (matches Python month_boundary 末 semantics).
fn try_month_xun(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let (year, rest) = resolve_optional_year_prefix(text, now)?;
    let rest = rest.trim_start_matches('的');
    let (month, rest) = parse_month_token(rest)?;
    let rest = rest
        .trim_start_matches('份')
        .trim_start_matches('的')
        .trim();
    let (from_day, to_day_kind) = match rest {
        "上旬" => (1u32, Xun::UpperFixed(10)),
        "中旬" => (11u32, Xun::UpperFixed(20)),
        "下旬" => (21u32, Xun::ToEom),
        "月底" | "底" => (0u32, Xun::Last5),
        _ => return None,
    };
    // Compute last day of that month.
    let first_of_next = if month == 12 {
        NaiveDate::from_ymd_opt(year + 1, 1, 1)?
    } else {
        NaiveDate::from_ymd_opt(year, month + 1, 1)?
    };
    let total = first_of_next.pred_opt()?.day();
    let (from_day, to_day) = match (from_day, to_day_kind) {
        (_, Xun::UpperFixed(d)) => (from_day, d.min(total)),
        (_, Xun::ToEom) => (from_day, total),
        (_, Xun::Last5) => (total.saturating_sub(4).max(25), total),
    };
    let from = NaiveDate::from_ymd_opt(year, month, from_day)?;
    let to = NaiveDate::from_ymd_opt(year, month, to_day)?;
    Some(TimeInfo {
        time_type: "time_span",
        start: from.and_hms_opt(0, 0, 0)?,
        end: to.and_hms_opt(23, 59, 59)?,
        definition: "blur",
        ..Default::default()
    })
}

enum Xun {
    UpperFixed(u32),
    ToEom,
    Last5,
}

/// `<year>首月/末月/第N个月/前N个月/后N个月`. Covers:
///   `2005年首月` → whole of January
///   `70年第8个月` → August whole
///   `五八年前七个月` → Jan-Jul
///   `二零二一年后三月` → Oct-Dec
fn try_year_ordinal_month(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let (year, rest) = resolve_optional_year_prefix(text, now)?;
    if year == now.year() && rest == text {
        // No year prefix → this function doesn't apply to bare forms.
        return None;
    }
    // 首月 / 末月.
    if rest == "首月" {
        let (start, end) = date_range(year, Some(1), None)?;
        return Some(TimeInfo {
            time_type: "time_span",
            start,
            end,
            definition: "accurate",
            ..Default::default()
        });
    }
    if rest == "末月" {
        let (start, end) = date_range(year, Some(12), None)?;
        return Some(TimeInfo {
            time_type: "time_span",
            start,
            end,
            definition: "accurate",
            ..Default::default()
        });
    }
    // 第N个月 (N-th month).
    static RE_NTH: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"^第\s*(\d+|[一二三四五六七八九十]+)\s*(?:个)?月$").unwrap());
    if let Some(caps) = RE_NTH.captures(rest) {
        let n_str = caps.get(1)?.as_str();
        let n: u32 = n_str.parse::<u32>().ok().or_else(|| cn_int(n_str))?;
        if (1..=12).contains(&n) {
            let (start, end) = date_range(year, Some(n), None)?;
            return Some(TimeInfo {
                time_type: "time_span",
                start,
                end,
                definition: "accurate",
                ..Default::default()
            });
        }
    }
    // 前N个月 / 后N个月.
    static RE_SPAN: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"^(前|后|首|末)\s*(\d+|[一二三四五六七八九十]+)\s*(?:个)?月$").unwrap()
    });
    if let Some(caps) = RE_SPAN.captures(rest) {
        let dir = caps.get(1)?.as_str();
        let n_str = caps.get(2)?.as_str();
        let n: u32 = n_str.parse::<u32>().ok().or_else(|| cn_int(n_str))?;
        if !(1..=12).contains(&n) {
            return None;
        }
        let (first_m, last_m) = match dir {
            "前" | "首" => (1u32, n),
            "后" | "末" => (13 - n, 12),
            _ => return None,
        };
        let first = NaiveDate::from_ymd_opt(year, first_m, 1)?;
        let end_year = if last_m == 12 { year + 1 } else { year };
        let end_month = if last_m == 12 { 1 } else { last_m + 1 };
        let last = NaiveDate::from_ymd_opt(end_year, end_month, 1)?.pred_opt()?;
        return Some(TimeInfo {
            time_type: "time_span",
            start: first.and_hms_opt(0, 0, 0)?,
            end: last.and_hms_opt(23, 59, 59)?,
            definition: "accurate",
            ..Default::default()
        });
    }
    None
}

/// `<year>?上半年/下半年` / `<year>?伊始` (== January).
fn try_half_year(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let (year, rest) = resolve_optional_year_prefix(text, now)?;
    let rest = rest.trim_start_matches('的');
    let (first_m, last_m) = match rest {
        "上半年" => (1u32, 6),
        "下半年" => (7u32, 12),
        "伊始" => (1u32, 1), // Python: 伊始 == the start month.
        _ => return None,
    };
    let first = NaiveDate::from_ymd_opt(year, first_m, 1)?;
    let next_year = if last_m == 12 { year + 1 } else { year };
    let next_month = if last_m == 12 { 1 } else { last_m + 1 };
    let last = NaiveDate::from_ymd_opt(next_year, next_month, 1)?.pred_opt()?;
    Some(TimeInfo {
        time_type: "time_span",
        start: first.and_hms_opt(0, 0, 0)?,
        end: last.and_hms_opt(23, 59, 59)?,
        definition: if rest == "伊始" { "blur" } else { "accurate" },
        ..Default::default()
    })
}

/// Relative-year + month (+ optional day + optional clock). Handles
/// `今年六月`, `明年3月份`, `去年3月3号`, `前年9月2号左右`, etc.
/// Day is optional (time_span whole month when omitted).
fn try_relative_year_month_day(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let (offset, rest) = limit_year_prefix(text)?;
    let rest = rest.trim_start_matches('的');

    let (month, rest) = parse_month_token(rest)?;
    // Strip optional `份` and `的`.
    let rest = rest.trim_start_matches('份').trim_start_matches('的');

    let year = now.year() + offset;
    if rest.is_empty() {
        // Whole month → time_span.
        let (start, end) = date_range(year, Some(month), None)?;
        return Some(TimeInfo {
            time_type: "time_span",
            start,
            end,
            definition: "accurate",
            ..Default::default()
        });
    }
    // Expect day (+ optional clock + optional 左右 modifier).
    let (day, after_day) = parse_day_token(rest)?;
    // Strip blur modifiers Python considers noise.
    let tail = after_day
        .trim_start_matches('左')
        .trim_start_matches('右')
        .trim();
    let (start, end) = date_range(year, Some(month), Some(day))?;
    apply_optional_clock(start, end, tail)
}

/// `<限定周> + <星期|周|礼拜>N` / `上个礼拜天` / `下周周六`. Resolves
/// the named week (本/上/下/上上/下下) to that week's Monday, then
/// offsets by the weekday number.
fn try_named_week_weekday(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    const WK_PREF: &[(&str, i32)] = &[
        ("本周", 0),
        ("这周", 0),
        ("这个周", 0),
        ("这个礼拜", 0),
        ("这个星期", 0),
        ("本星期", 0),
        ("本礼拜", 0),
        ("上周", -1),
        ("上个周", -1),
        ("上一周", -1),
        ("上个星期", -1),
        ("上个礼拜", -1),
        ("上礼拜", -1),
        ("上星期", -1),
        ("上上周", -2),
        ("下周", 1),
        ("下个周", 1),
        ("下一周", 1),
        ("下个星期", 1),
        ("下个礼拜", 1),
        ("下礼拜", 1),
        ("下星期", 1),
        ("下下周", 2),
    ];
    let mut best: Option<(i32, &str)> = None;
    for (pref, off) in WK_PREF {
        if let Some(rest) = text.strip_prefix(*pref) {
            match best {
                Some((_, r)) if r.len() < rest.len() => {}
                _ => best = Some((*off, rest)),
            }
        }
    }
    let (week_offset, rest) = best?;
    let rest = rest.trim_start_matches('的');
    // Optional redundant leading `周/星期/礼拜`.
    let rest = rest
        .strip_prefix("星期")
        .or_else(|| rest.strip_prefix("礼拜"))
        .or_else(|| rest.strip_prefix('周'))
        .unwrap_or(rest);
    let weekday: u32 = match rest {
        "一" | "1" => 1,
        "二" | "2" => 2,
        "三" | "3" => 3,
        "四" | "4" => 4,
        "五" | "5" => 5,
        "六" | "6" => 6,
        "日" | "天" | "7" | "日早上" | "天早上" => 7,
        _ => return None,
    };
    // Find Monday of this week, offset by week_offset, add (weekday - 1).
    let today = now.date();
    let today_wday = today.weekday().num_days_from_monday() as i64;
    let this_monday = today - chrono::Duration::days(today_wday);
    let target_monday = this_monday + chrono::Duration::days((week_offset as i64) * 7);
    let target_date = target_monday + chrono::Duration::days((weekday as i64) - 1);
    Some(TimeInfo {
        time_type: "time_point",
        start: target_date.and_hms_opt(0, 0, 0)?,
        end: target_date.and_hms_opt(23, 59, 59)?,
        definition: "accurate",
        ..Default::default()
    })
}

/// `<M月>第N周` / `<Y年M月>第N周` / `<限定月>第N周` /
/// `<相对年M月>第N周`. Mirrors Python's `self.month_week_pattern`.
/// Week boundaries: week 1 starts on the first Monday of the month
/// (Python default).
fn try_month_week_ordinal(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    // Resolve optional year prefix (Arabic / Chinese / relative words).
    let (year, after_year) = resolve_optional_year_prefix(text, now)?;

    // Optional `的` between year and month.
    let after_year = after_year.trim_start_matches('的');

    // Resolve optional limit-month prefix (本/上/下月 etc.) OR a literal
    // `M月` / `M月份`. Produces (month, offset_years_applied, rest).
    // `offset_years_applied` handles the case where relative-month wraps.
    let (month, rest_after_month) =
        if let Some((m, rest)) = try_resolve_limit_month(after_year, now) {
            (m, rest)
        } else if let Some((m, rest)) = parse_month_token(after_year) {
            // Month-份 suffix is optional.
            let rest = rest.trim_start_matches('份');
            (m, rest)
        } else {
            return None;
        };

    // Optional `的` between month and 第N周.
    let rest = rest_after_month.trim_start_matches('的');

    // `第N周` — N in 1..=5.
    static RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^第\s*(\d+|[一二三四五])\s*周$").unwrap());
    let caps = RE.captures(rest)?;
    let n_str = caps.get(1)?.as_str();
    let n: u32 = n_str.parse::<u32>().ok().or_else(|| cn_int(n_str))?;
    if !(1..=5).contains(&n) {
        return None;
    }

    // First Monday of the month.
    let first = NaiveDate::from_ymd_opt(year, month, 1)?;
    let wday = first.weekday().num_days_from_monday(); // 0..6
    let first_monday = first + chrono::Duration::days(((7 - wday) % 7) as i64);
    let week_start = first_monday + chrono::Duration::days(((n - 1) * 7) as i64);
    let week_end = week_start + chrono::Duration::days(6);

    Some(TimeInfo {
        time_type: "time_span",
        start: week_start.and_hms_opt(0, 0, 0)?,
        end: week_end.and_hms_opt(23, 59, 59)?,
        definition: "accurate",
        ..Default::default()
    })
}

/// Strip an optional year prefix (`2024年`, `二零二四年`, `今年`, `去年` …)
/// and return `(resolved_year, rest_after_year)`. When no prefix is
/// present, returns `(now.year(), text)` so the caller can still proceed.
fn resolve_optional_year_prefix<'a>(text: &'a str, now: NaiveDateTime) -> Option<(i32, &'a str)> {
    // Relative-year prefix (longest match).
    const REL: &[(&str, i32)] = &[
        ("大前年", -3),
        ("大后年", 3),
        ("明年", 1),
        ("次年", 1),
        ("后年", 2),
        ("去年", -1),
        ("前年", -2),
        ("今年", 0),
        ("本年", 0),
        ("这年", 0),
    ];
    let mut best: Option<(i32, &str)> = None;
    for (pref, off) in REL {
        if let Some(rest) = text.strip_prefix(*pref) {
            match best {
                Some((_, r)) if r.len() < rest.len() => {}
                _ => best = Some((*off, rest)),
            }
        }
    }
    if let Some((off, rest)) = best {
        return Some((now.year() + off, rest));
    }
    // Arabic year.
    static RE_Y: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(\d{2,4})\s*年").unwrap());
    if let Some(caps) = RE_Y.captures(text) {
        let year = parse_year(caps.get(1)?.as_str())?;
        let rest = &text[caps.get(0)?.end()..];
        return Some((year, rest));
    }
    // Chinese year.
    static RE_CN: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"^([零〇一二三四五六七八九十两]+)\s*年").unwrap());
    if let Some(caps) = RE_CN.captures(text) {
        let year = parse_chinese_year_digits(caps.get(1)?.as_str())?;
        let rest = &text[caps.get(0)?.end()..];
        return Some((year, rest));
    }
    // No year prefix — inherit current year.
    Some((now.year(), text))
}

/// Resolve `本月` / `上个月` / `下月` / `当月` etc. returning `(month_1_12,
/// rest_after_month)` inheriting year from `now`. Returns None if there
/// is no limit-month prefix; use `parse_month_token` for literal `M月`.
fn try_resolve_limit_month<'a>(text: &'a str, now: NaiveDateTime) -> Option<(u32, &'a str)> {
    const MONTH_PREF: &[(&str, i32)] = &[
        ("本月", 0),
        ("当月", 0),
        ("这个月", 0),
        ("这月", 0),
        ("下个月", 1),
        ("下一个月", 1),
        ("下月", 1),
        ("上个月", -1),
        ("上一个月", -1),
        ("上月", -1),
    ];
    let mut best: Option<(i32, &str)> = None;
    for (pref, off) in MONTH_PREF {
        if let Some(rest) = text.strip_prefix(*pref) {
            match best {
                Some((_, r)) if r.len() < rest.len() => {}
                _ => best = Some((*off, rest)),
            }
        }
    }
    let (offset, rest) = best?;
    let mut m = now.month() as i32 + offset;
    while m <= 0 {
        m += 12;
    }
    while m > 12 {
        m -= 12;
    }
    Some((m as u32, rest))
}

/// Relative-year + year-boundary: `明年初` / `去年底` / `今年年末` /
/// `次年年初` / `年底` / `年初` (bare, inherits current year).
fn try_relative_year_blur_boundary(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    const YEAR_OFFSETS: &[(&str, i32)] = &[
        ("明年", 1),
        ("次年", 1),
        ("今年", 0),
        ("本年", 0),
        ("这年", 0),
        ("去年", -1),
        ("前年", -2),
        ("大前年", -3),
        ("后年", 2),
        ("大后年", 3),
    ];
    // Find longest matching year prefix (or none → current year).
    let (year_offset, after_year) = {
        let mut best: Option<(i32, &str)> = None;
        for (pref, off) in YEAR_OFFSETS {
            if let Some(rest) = text.strip_prefix(*pref) {
                match best {
                    Some((_, r)) if r.len() < rest.len() => {}
                    _ => best = Some((*off, rest)),
                }
            }
        }
        match best {
            Some(x) => x,
            None => (0, text),
        }
    };
    // Optional leading `年` (e.g. "明年年初" = 明年 + 年初).
    let after_year = after_year.strip_prefix('年').unwrap_or(after_year);
    // Must match exactly a boundary token.
    let pos = match after_year {
        "初" | "头" => "初",
        "末" | "底" | "尾" => "末",
        "中" => "中",
        _ => return None,
    };
    let year = now.year() + year_offset;
    let (first_month, last_month) = match pos {
        "初" => (1u32, 2),
        "末" => (11u32, 12),
        "中" => (6u32, 7),
        _ => return None,
    };
    let first = NaiveDate::from_ymd_opt(year, first_month, 1)?;
    let next = if last_month == 12 {
        NaiveDate::from_ymd_opt(year + 1, 1, 1)?
    } else {
        NaiveDate::from_ymd_opt(year, last_month + 1, 1)?
    };
    let last = next.pred_opt()?;
    Some(TimeInfo {
        time_type: "time_span",
        start: first.and_hms_opt(0, 0, 0)?,
        end: last.and_hms_opt(23, 59, 59)?,
        definition: "blur",
        ..Default::default()
    })
}

/// Pattern #15 — `YYYY年初` / `YYYY年末` / `YYYY年底` / `YYYY年中` /
/// Chinese-year variants, plus the bare `年底` / `年初` suffix alone
/// (redundant prefix `年`, like `YYYY年年初`).
fn try_year_blur_boundary(text: &str) -> Option<TimeInfo> {
    static RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"^(\d{2,4})\s*年\s*(?:年)?(初|末|底|中|头|尾)$").unwrap());
    static RE_CN: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"^([零〇一二三四五六七八九十两]+)\s*年\s*(?:年)?(初|末|底|中|头|尾)$").unwrap()
    });
    let (year, pos) = if let Some(caps) = RE.captures(text) {
        (
            parse_year(caps.get(1)?.as_str())?,
            caps.get(2)?.as_str().to_string(),
        )
    } else if let Some(caps) = RE_CN.captures(text) {
        (
            parse_chinese_year_digits(caps.get(1)?.as_str())?,
            caps.get(2)?.as_str().to_string(),
        )
    } else {
        return None;
    };
    let pos = pos.as_str();
    let (first_month, last_month) = match pos {
        "初" | "头" => (1u32, 2),          // Jan-Feb
        "末" | "底" | "尾" => (11u32, 12), // Nov-Dec
        "中" => (6u32, 7),                 // Jun-Jul
        _ => return None,
    };
    let first = NaiveDate::from_ymd_opt(year, first_month, 1)?;
    let next = if last_month == 12 {
        NaiveDate::from_ymd_opt(year + 1, 1, 1)?
    } else {
        NaiveDate::from_ymd_opt(year, last_month + 1, 1)?
    };
    let last = next.pred_opt()?;
    Some(TimeInfo {
        time_type: "time_span",
        start: first.and_hms_opt(0, 0, 0)?,
        end: last.and_hms_opt(23, 59, 59)?,
        definition: "blur",
        ..Default::default()
    })
}

/// Pattern #18 — `本/上/下(个)?月<day>(日|号)` inheriting computed year+month.
fn try_limit_month_day(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    const OFFSETS: &[(&str, i32)] = &[
        ("本月", 0),
        ("这个月", 0),
        ("这月", 0),
        ("下个月", 1),
        ("下一个月", 1),
        ("下月", 1),
        ("上个月", -1),
        ("上一个月", -1),
        ("上月", -1),
    ];
    let mut matched: Option<(i32, &str)> = None;
    for (prefix, off) in OFFSETS {
        if let Some(rest) = text.strip_prefix(*prefix) {
            match matched {
                Some((_, r)) if r.len() < rest.len() => {}
                _ => matched = Some((*off, rest)),
            }
        }
    }
    let (offset, rest) = matched?;
    let rest = rest.trim_start_matches(['的', ' ']);
    let (day, tail) = parse_day_token(rest)?;
    if !tail.trim().is_empty() {
        return None;
    }

    // Compute target year/month.
    let mut y = now.year();
    let mut m = now.month() as i32 + offset;
    while m <= 0 {
        m += 12;
        y -= 1;
    }
    while m > 12 {
        m -= 12;
        y += 1;
    }
    let (start, end) = date_range(y, Some(m as u32), Some(day))?;
    Some(TimeInfo {
        time_type: "time_point",
        start,
        end,
        definition: "accurate",
        ..Default::default()
    })
}

/// Same as `try_bare_month_day` but uses a caller-supplied year (for
/// try_date_range to inherit the LHS's year on the RHS).
fn try_bare_month_day_with_year(text: &str, year: i32, _now: NaiveDateTime) -> Option<TimeInfo> {
    if let Some((month, rest)) = parse_month_token(text) {
        if let Some((day, tail)) = parse_day_token(rest) {
            let (start, end) = date_range(year, Some(month), Some(day))?;
            return apply_optional_clock(start, end, tail.trim());
        }
    }
    None
}

/// Python parity — `M月D日` / `M月D日<clock>` / `M月D` (no year).
/// Inherits year from `now`. Matches Python `year_month_day_pattern`'s
/// middle branch (`bracket(MONTH_STRING), bracket_absence(DAY_STRING)`).
fn try_bare_month_day(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    // Text must begin with month token; reject if any preceding junk.
    // Accept 1-2 digit or Chinese numeral month, followed by `月`.
    if let Some((month, rest)) = parse_month_token(text) {
        // Python convention: for bare (no-year) forms, require an
        // explicit day marker (日/号) when the day is Chinese numerals —
        // otherwise `二月十五` is ambiguous and should fall through to
        // lunar parsing. Allow Arabic digits without marker.
        let rest_trimmed = rest.trim_start();
        let day_is_arabic = rest_trimmed
            .chars()
            .next()
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false);
        let has_day_marker = rest_trimmed.contains('日') || rest_trimmed.contains('号');
        if !day_is_arabic && !has_day_marker {
            // Let lunar (or later Gregorian with full year) handle it.
        } else if let Some((day, tail)) = parse_day_token(rest) {
            let year = now.year();
            let (start, end) = date_range(year, Some(month), Some(day))?;
            return apply_optional_clock(start, end, tail.trim());
        }
    }
    // `09-01` / `9/1` / `09.01` — MM-DD with ASCII separator, no year.
    // Optional clock tail. Must distinguish from HH:MM (4+ digits with
    // colon), hence require the separator NOT to be `:` and require a
    // space before the clock tail when present.
    static MD_DASH: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"^(\d{1,2})[\-/\.](\d{1,2})(\s+.+|[\-\s]*\d{1,2}[:：].*)?$").unwrap()
    });
    if let Some(caps) = MD_DASH.captures(text) {
        let month: u32 = caps.get(1)?.as_str().parse().ok()?;
        let day: u32 = caps.get(2)?.as_str().parse().ok()?;
        if (1..=12).contains(&month) && (1..=31).contains(&day) {
            let year = now.year();
            let (start, end) = date_range(year, Some(month), Some(day))?;
            let tail = caps.get(3).map(|m| m.as_str()).unwrap_or("");
            return apply_optional_clock(start, end, tail.trim());
        }
    }

    // `6·30` / `10·1` — middle-dot month·day, no year. Python parity:
    // see jionlp tests `'6·30'` → time_point.
    static MD_DOT: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(\d{1,2})·(\d{1,2})$").unwrap());
    if let Some(caps) = MD_DOT.captures(text) {
        let month: u32 = caps.get(1)?.as_str().parse().ok()?;
        let day: u32 = caps.get(2)?.as_str().parse().ok()?;
        if (1..=12).contains(&month) && (1..=31).contains(&day) {
            let year = now.year();
            let (start, end) = date_range(year, Some(month), Some(day))?;
            return Some(TimeInfo {
                time_type: "time_point",
                start,
                end,
                definition: "accurate",
                ..Default::default()
            });
        }
    }
    None
}

/// Python parity — `D日H时` / `D号H点` (no year, no month) — used for
/// the right-hand side of spans like `本月10日至12日16时`. Inherits both
/// year and month from `now`.
fn try_bare_day_with_clock(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let (day, rest) = parse_day_token(text)?;
    // Must be strictly day+clock — no empty tail, no month/year markers.
    let rest = rest.trim();
    if rest.is_empty() {
        return None;
    }
    if rest.contains('月') || rest.contains('年') {
        return None;
    }
    let (year, month) = (now.year(), now.month());
    let (start, end) = date_range(year, Some(month), Some(day))?;
    apply_optional_clock(start, end, rest)
}

/// Parse a leading month token (1..=12) as either Arabic digits (1 or 2)
/// followed by `月`, or Chinese numeral month followed by `月`. Returns
/// `(month, rest_after_月)`.
fn parse_month_token(s: &str) -> Option<(u32, &str)> {
    let s = s.trim_start();
    // Try Arabic digits (up to 2, bounded by `月`).
    let bytes = s.as_bytes();
    let mut end = 0usize;
    while end < bytes.len() && bytes[end].is_ascii_digit() && end < 2 {
        end += 1;
    }
    if end > 0 {
        let m: u32 = s[..end].parse().ok()?;
        if !(1..=12).contains(&m) {
            return None;
        }
        let rest = &s[end..];
        if let Some(rest) = rest.strip_prefix('月') {
            return Some((m, rest));
        }
    }
    // Chinese numeral month: 一..九, 十, 十一, 十二.
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    for take in [2usize, 1] {
        if chars.len() < take {
            continue;
        }
        let end_byte = if chars.len() == take {
            s.len()
        } else {
            chars[take].0
        };
        let chunk = &s[..end_byte];
        if let Some(n) = cn_int(chunk) {
            if (1..=12).contains(&n) {
                let rest = &s[end_byte..];
                if let Some(rest) = rest.strip_prefix('月') {
                    return Some((n, rest));
                }
            }
        }
    }
    None
}

/// Pattern #16 — `本/上/下月末` / `本月初/中`.
fn try_limit_month_boundary(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    const OFFSETS: &[(&str, i32)] = &[
        ("本月", 0),
        ("这月", 0),
        ("这个月", 0),
        ("下个月", 1),
        ("下月", 1),
        ("下一个月", 1),
        ("上个月", -1),
        ("上月", -1),
        ("上一个月", -1),
    ];
    let mut matched: Option<(i32, &str)> = None;
    for (prefix, off) in OFFSETS {
        if let Some(rest) = text.strip_prefix(*prefix) {
            match matched {
                Some((_, r)) if r.len() < rest.len() => {}
                _ => matched = Some((*off, rest)),
            }
        }
    }
    let (offset, rest) = matched?;
    let rest = rest.trim();
    let portion = match rest {
        "初" => 0i32,
        "中" => 1,
        "末" | "底" => 2,
        _ => return None,
    };

    let mut y = now.year();
    let mut m = now.month() as i32 + offset;
    while m <= 0 {
        m += 12;
        y -= 1;
    }
    while m > 12 {
        m -= 12;
        y += 1;
    }
    let next_first = if m == 12 {
        NaiveDate::from_ymd_opt(y + 1, 1, 1)?
    } else {
        NaiveDate::from_ymd_opt(y, (m + 1) as u32, 1)?
    };
    let last = next_first.pred_opt()?;
    let total = last.day();
    // Python convention (accurate, narrow windows):
    //   初 = days 1-5, 中 = 15-20, 末/底 = last 5 days.
    let (from_day, to_day) = match portion {
        0 => (1u32, total.min(5)),
        1 => (15u32, total.min(20)),
        _ => ((total.saturating_sub(4)).max(25), total),
    };
    let from = NaiveDate::from_ymd_opt(y, m as u32, from_day)?;
    let to = NaiveDate::from_ymd_opt(y, m as u32, to_day)?;
    Some(TimeInfo {
        time_type: "time_point",
        start: from.and_hms_opt(0, 0, 0)?,
        end: to.and_hms_opt(23, 59, 59)?,
        definition: "accurate",
        ..Default::default()
    })
}

/// Pattern #20 — century / decade / period. Python conventions:
///   * `18世纪` → 1700–1799 (hundred-mark convention, not strict
///     "1701-1800").
///   * `上世纪` → previous century relative to `now`.
///   * `N世纪M十年代[前期|中期|后期|末期|初|末]` → decade with sub-period.
///   * `N世纪初/末` → first/last 20 years of the century.
fn try_century(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    fn split_modifier(s: &str) -> (&str, Option<&'static str>) {
        // Longest suffix first so `末期` beats `末`.
        let mods: [(&str, &str); 7] = [
            ("前期", "early"),
            ("中期", "mid"),
            ("后期", "late"),
            ("末期", "late"),
            ("初期", "early"),
            ("初", "very_early"),
            ("末", "very_late"),
        ];
        for (suf, tag) in mods {
            if let Some(rest) = s.strip_suffix(suf) {
                return (rest, Some(tag));
            }
        }
        (s, None)
    }

    // Resolve optional relative-century prefix OR numeric century.
    let now_century = (now.year() - 1) / 100 + 1;
    let (century, after_prefix): (i32, &str) = if let Some(rest) = text.strip_prefix("上世纪") {
        (now_century - 1, rest)
    } else if let Some(rest) = text
        .strip_prefix("本世纪")
        .or_else(|| text.strip_prefix("这世纪"))
    {
        (now_century, rest)
    } else if let Some(rest) = text.strip_prefix("下世纪") {
        (now_century + 1, rest)
    } else {
        static RE_C: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"^(\d{1,2}|[一二两三四五六七八九十]+)\s*世纪").unwrap());
        let caps = RE_C.captures(text)?;
        let c_str = caps.get(1)?.as_str();
        let c: i32 = c_str
            .parse::<i32>()
            .ok()
            .or_else(|| cn_int(c_str).map(|n| n as i32))?;
        if !(1..=30).contains(&c) {
            return None;
        }
        (c, &text[caps.get(0)?.end()..])
    };

    if !(1..=30).contains(&century) {
        return None;
    }
    let c_start = (century - 1) * 100; // 18世纪 → 1700

    let (body, modifier) = split_modifier(after_prefix);

    if body.is_empty() {
        let (y_first, y_last) = match modifier {
            Some("very_early") => (c_start, c_start + 19),
            Some("very_late") => (c_start + 80, c_start + 99),
            Some("early") => (c_start, c_start + 29),
            Some("mid") => (c_start + 30, c_start + 69),
            Some("late") => (c_start + 70, c_start + 99),
            _ => (c_start, c_start + 99),
        };
        let first = NaiveDate::from_ymd_opt(y_first, 1, 1)?;
        let last = NaiveDate::from_ymd_opt(y_last, 12, 31)?;
        return Some(TimeInfo {
            time_type: "time_span",
            start: first.and_hms_opt(0, 0, 0)?,
            end: last.and_hms_opt(23, 59, 59)?,
            definition: "blur",
            ..Default::default()
        });
    }

    // Decade: `M十年代` (Chinese) or `\d+年代` (Arabic).
    static RE_D: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"^(十|[一二三四五六七八九]十|\d+)\s*年代$").unwrap());
    let caps = RE_D.captures(body)?;
    let d_str = caps.get(1)?.as_str();
    let decade: i32 = if let Ok(n) = d_str.parse::<i32>() {
        n
    } else if d_str == "十" {
        10
    } else if let Some(prefix) = d_str.strip_suffix('十') {
        (cn_int(prefix)? as i32) * 10
    } else {
        return None;
    };
    if !(0..=90).contains(&decade) {
        return None;
    }
    let d_start = c_start + decade;

    let (y_first, y_last) = match modifier {
        Some("very_early") | Some("early") => (d_start, d_start + 2),
        Some("mid") => (d_start + 3, d_start + 6),
        Some("late") | Some("very_late") => (d_start + 7, d_start + 9),
        _ => (d_start, d_start + 9),
    };
    let first = NaiveDate::from_ymd_opt(y_first, 1, 1)?;
    let last = NaiveDate::from_ymd_opt(y_last, 12, 31)?;
    Some(TimeInfo {
        time_type: "time_span",
        start: first.and_hms_opt(0, 0, 0)?,
        end: last.and_hms_opt(23, 59, 59)?,
        definition: "blur",
        ..Default::default()
    })
}

/// Parse Chinese numerals up to 万 (10 000). Used for century / blur year.
fn parse_chinese_number(s: &str) -> Option<i32> {
    // Simple handler for up to 4 digits: supports forms like 一百二十三,
    // 三千, 两万. Not a full classical parser — good enough for the time
    // parser's numeric prefixes.
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    // Fast path: all small-digit characters ⇒ use cn_int.
    if let Some(n) = cn_int(s) {
        return Some(n as i32);
    }

    let digit = |c: char| -> Option<i32> {
        match c {
            '零' | '〇' => Some(0),
            '一' => Some(1),
            '二' | '两' => Some(2),
            '三' => Some(3),
            '四' => Some(4),
            '五' => Some(5),
            '六' => Some(6),
            '七' => Some(7),
            '八' => Some(8),
            '九' => Some(9),
            _ => None,
        }
    };
    let unit = |c: char| -> Option<i32> {
        match c {
            '十' => Some(10),
            '百' => Some(100),
            '千' => Some(1000),
            '万' => Some(10_000),
            _ => None,
        }
    };

    let mut total = 0i32;
    let mut section = 0i32;
    let mut pending = 0i32;
    let mut has_pending = false;
    let mut last_was_ten_thousand = false;

    for c in s.chars() {
        if let Some(d) = digit(c) {
            pending = d;
            has_pending = true;
        } else if let Some(u) = unit(c) {
            if u == 10_000 {
                let block = if has_pending {
                    section + pending
                } else {
                    section
                };
                total += block * 10_000;
                section = 0;
                pending = 0;
                has_pending = false;
                last_was_ten_thousand = true;
            } else {
                let base = if has_pending { pending } else { 1 };
                section += base * u;
                pending = 0;
                has_pending = false;
                last_was_ten_thousand = false;
            }
        } else {
            return None;
        }
    }
    let final_block = section + if has_pending { pending } else { 0 };
    if last_was_ten_thousand && final_block == 0 {
        Some(total)
    } else {
        Some(total + final_block)
    }
}

/// Pattern #57, #55, #56 — named special phrases mapped to fixed spans.
/// `现在/此时/此刻/目前` → now (one-second window);
/// `全天` → today; `全月` → this month; `全年` → this year;
/// `今明两天/明后两天/前后两天` → 2-day span.
fn try_special_phrases(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    match text {
        "现在" | "此时" | "此刻" | "目前" | "眼下" | "如今" => {
            let dt = now;
            Some(TimeInfo {
                time_type: "time_point",
                start: dt,
                end: dt,
                definition: "accurate",
                ..Default::default()
            })
        }
        "全天" | "一整天" | "整天" => {
            let d = now.date();
            Some(TimeInfo {
                time_type: "time_span",
                start: d.and_hms_opt(0, 0, 0)?,
                end: d.and_hms_opt(23, 59, 59)?,
                definition: "accurate",
                ..Default::default()
            })
        }
        "全月" | "整月" | "一整月" => {
            let (s, e) = period_range(NamedPeriod::Month, now.date(), 0)?;
            Some(TimeInfo {
                time_type: "time_span",
                start: s,
                end: e,
                definition: "accurate",
                ..Default::default()
            })
        }
        "全年" | "整年" | "一整年" => {
            let (s, e) = period_range(NamedPeriod::Year, now.date(), 0)?;
            Some(TimeInfo {
                time_type: "time_span",
                start: s,
                end: e,
                definition: "accurate",
                ..Default::default()
            })
        }
        "今明两天" => {
            let d0 = now.date();
            let d1 = d0 + Duration::days(1);
            Some(TimeInfo {
                time_type: "time_span",
                start: d0.and_hms_opt(0, 0, 0)?,
                end: d1.and_hms_opt(23, 59, 59)?,
                definition: "accurate",
                ..Default::default()
            })
        }
        "明后两天" => {
            let d0 = now.date() + Duration::days(1);
            let d1 = d0 + Duration::days(1);
            Some(TimeInfo {
                time_type: "time_span",
                start: d0.and_hms_opt(0, 0, 0)?,
                end: d1.and_hms_opt(23, 59, 59)?,
                definition: "accurate",
                ..Default::default()
            })
        }
        "前后两天" => {
            let d0 = now.date() - Duration::days(1);
            let d1 = now.date() + Duration::days(1);
            Some(TimeInfo {
                time_type: "time_span",
                start: d0.and_hms_opt(0, 0, 0)?,
                end: d1.and_hms_opt(23, 59, 59)?,
                definition: "accurate",
                ..Default::default()
            })
        }
        _ => None,
    }
}

/// Pattern #40 — standalone weekday `周一` / `星期五` → this week's occurrence
/// (current day if matches, else next).
fn try_standalone_weekday(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let (weekday, rest) = match_weekday(text)?;
    if !rest.is_empty() {
        return None;
    }
    let today = now.date();
    let cur = today.weekday().num_days_from_monday();
    let delta = (weekday + 7 - cur) % 7;
    let target = today + Duration::days(delta as i64);
    Some(TimeInfo {
        time_type: "time_point",
        start: target.and_hms_opt(0, 0, 0)?,
        end: target.and_hms_opt(23, 59, 59)?,
        definition: "accurate",
        ..Default::default()
    })
}

/// Pattern #41 — `本周一` / `上周五` / `下周三`.
fn try_named_weekday(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    // Prefix chart: "本/这" → 0, "上" → -1, "下" → 1. The body after the
    // prefix must start with 周X/星期X, so we strip only the scope word.
    const OFFSETS: &[(&str, i32)] = &[
        ("本", 0),
        ("这", 0),
        ("上个", -1),
        ("上", -1),
        ("下个", 1),
        ("下", 1),
    ];
    let mut matched: Option<(i32, &str)> = None;
    for (prefix, off) in OFFSETS {
        if let Some(rest) = text.strip_prefix(*prefix) {
            match matched {
                Some((_, r)) if r.len() < rest.len() => {}
                _ => matched = Some((*off, rest)),
            }
        }
    }
    let (offset, rest) = matched?;
    let (weekday, rest) = match_weekday(rest)?;
    if !rest.is_empty() {
        return None;
    }
    let today = now.date();
    let dow = today.weekday().num_days_from_monday() as i64;
    let this_mon = today - Duration::days(dow);
    let target_mon = this_mon + Duration::days(offset as i64 * 7);
    let target = target_mon + Duration::days(weekday as i64);
    Some(TimeInfo {
        time_type: "time_point",
        start: target.and_hms_opt(0, 0, 0)?,
        end: target.and_hms_opt(23, 59, 59)?,
        definition: "accurate",
        ..Default::default()
    })
}

/// Pattern #47 — `YYYY年第N周` / `YYYY年第N个星期`. ISO-ish: week 1 starts
/// on the first Monday of the year.
fn try_year_week(text: &str) -> Option<TimeInfo> {
    static RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"^(\d{2,4})\s*年\s*第\s*(\d+|[一二两三四五六七八九十]+)\s*(?:个)?\s*(?:周|星期)$",
        )
        .unwrap()
    });
    let caps = RE.captures(text)?;
    let year = parse_year(caps.get(1)?.as_str())?;
    let n_str = caps.get(2)?.as_str();
    let n: i32 = n_str
        .parse::<i32>()
        .ok()
        .or_else(|| parse_chinese_number(n_str))?;
    if !(1..=53).contains(&n) {
        return None;
    }
    let jan1 = NaiveDate::from_ymd_opt(year, 1, 1)?;
    let first_mon_offset = (7 - jan1.weekday().num_days_from_monday()) % 7;
    let first_monday = jan1 + Duration::days(first_mon_offset as i64);
    let start = first_monday + Duration::days((n as i64 - 1) * 7);
    let end = start + Duration::days(6);
    Some(TimeInfo {
        time_type: "time_span",
        start: start.and_hms_opt(0, 0, 0)?,
        end: end.and_hms_opt(23, 59, 59)?,
        definition: "accurate",
        ..Default::default()
    })
}

/// Pattern #43-#46 — `YYYY年M月的第N个周K` / `M月第N个周K` / `本月第N个周K`.
fn try_nth_weekday_in_month(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    // Three forms share a common tail: "第N个周K" / "第N个星期K".
    static TAIL_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"第\s*(\d+|[一二两三四五六七八九十]+)\s*个\s*(周[一二三四五六日天]|星期[一二三四五六日天])$",
        )
        .unwrap()
    });
    let caps = TAIL_RE.captures(text)?;
    let n_str = caps.get(1)?.as_str();
    let nth: u32 = n_str.parse::<u32>().ok().or_else(|| cn_int(n_str))?;
    if !(1..=5).contains(&nth) {
        return None;
    }
    let wd_str = caps.get(2)?.as_str();
    let (weekday, _) = match_weekday(wd_str)?;
    let head = &text[..caps.get(0)?.start()];
    let head = head.trim_end_matches('的').trim();

    // Three head shapes:
    //   "YYYY年M月"   → explicit year+month
    //   "M月"         → current year, explicit month
    //   "本月"/"下个月" → computed y+m relative to now
    let (year, month) = if let Some(caps2) = {
        static H1: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"^(\d{2,4})\s*年\s*(\d{1,2})\s*月$").unwrap());
        H1.captures(head)
    } {
        (
            parse_year(caps2.get(1)?.as_str())?,
            caps2.get(2)?.as_str().parse::<u32>().ok()?,
        )
    } else if let Some(caps2) = {
        static H2: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(\d{1,2})\s*月$").unwrap());
        H2.captures(head)
    } {
        (now.year(), caps2.get(1)?.as_str().parse::<u32>().ok()?)
    } else {
        const OFFSETS: &[(&str, i32)] = &[
            ("本月", 0),
            ("这个月", 0),
            ("这月", 0),
            ("下个月", 1),
            ("下月", 1),
            ("下一个月", 1),
            ("上个月", -1),
            ("上月", -1),
            ("上一个月", -1),
        ];
        let mut found: Option<i32> = None;
        for (p, off) in OFFSETS {
            if head == *p {
                found = Some(*off);
                break;
            }
        }
        let off = found?;
        let mut y = now.year();
        let mut m = now.month() as i32 + off;
        while m <= 0 {
            m += 12;
            y -= 1;
        }
        while m > 12 {
            m -= 12;
            y += 1;
        }
        (y, m as u32)
    };

    let target = nth_weekday_of_month(year, month, weekday, nth)?;
    Some(TimeInfo {
        time_type: "time_point",
        start: target.and_hms_opt(0, 0, 0)?,
        end: target.and_hms_opt(23, 59, 59)?,
        definition: "accurate",
        ..Default::default()
    })
}

/// Pattern #52/#53 — `YYYY年第N天` (365-day nth day) / bare `第N天` in
/// current year.
fn try_year_day_ordinal(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    static RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"^(?:(\d{2,4})\s*年\s*)?第\s*(\d+|[一二两三四五六七八九十百千]+)\s*天$")
            .unwrap()
    });
    let caps = RE.captures(text)?;
    let year = match caps.get(1) {
        Some(m) => parse_year(m.as_str())?,
        None => now.year(),
    };
    let n_str = caps.get(2)?.as_str();
    let n: i32 = n_str
        .parse::<i32>()
        .ok()
        .or_else(|| parse_chinese_number(n_str))?;
    if !(1..=366).contains(&n) {
        return None;
    }
    let jan1 = NaiveDate::from_ymd_opt(year, 1, 1)?;
    let d = jan1 + Duration::days((n - 1) as i64);
    if d.year() != year {
        return None;
    }
    Some(TimeInfo {
        time_type: "time_point",
        start: d.and_hms_opt(0, 0, 0)?,
        end: d.and_hms_opt(23, 59, 59)?,
        definition: "accurate",
        ..Default::default()
    })
}

/// Pattern #8 — `前两天 / 前三个月 / 前两年 / 前一周`.
fn try_super_blur_ymd(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    static RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"^(前|近|过去|未来|这)\s*(\d+|[一二两三四五六七八九十]+)\s*(?:个\s*)?(天|日|周|星期|月|年|季度)$",
        )
        .unwrap()
    });
    let caps = RE.captures(text)?;
    let dir = caps.get(1)?.as_str();
    let n_str = caps.get(2)?.as_str();
    let n: i32 = n_str
        .parse::<i32>()
        .ok()
        .or_else(|| cn_int(n_str).map(|v| v as i32))?;
    let unit = caps.get(3)?.as_str();
    let sign: i32 = match dir {
        "前" | "近" | "过去" | "这" => -1,
        "未来" => 1,
        _ => return None,
    };
    // `这一X` is idiomatic "this X" (whole calendar unit), not
    // "past 1 X". Hand it off to try_named_period so `这一年` returns
    // the full year and `这一月` returns the full month.
    if dir == "这" && n == 1 {
        return None;
    }
    let today = now.date();
    let (start, end) = match unit {
        "天" | "日" => {
            let a = today + Duration::days(sign as i64 * (n - 1) as i64);
            let b = today;
            let (s, e) = if a <= b { (a, b) } else { (b, a) };
            (s.and_hms_opt(0, 0, 0)?, e.and_hms_opt(23, 59, 59)?)
        }
        "周" | "星期" => {
            let a = today + Duration::days(sign as i64 * (n * 7 - 1) as i64);
            let b = today;
            let (s, e) = if a <= b { (a, b) } else { (b, a) };
            (s.and_hms_opt(0, 0, 0)?, e.and_hms_opt(23, 59, 59)?)
        }
        "月" => {
            let a_dt = add_months(now, sign * (n - 1))?;
            let a = a_dt.date();
            let b = today;
            let (s, e) = if a <= b { (a, b) } else { (b, a) };
            (s.and_hms_opt(0, 0, 0)?, e.and_hms_opt(23, 59, 59)?)
        }
        "年" => {
            let a =
                NaiveDate::from_ymd_opt(today.year() + sign * (n - 1), today.month(), today.day())?;
            let b = today;
            let (s, e) = if a <= b { (a, b) } else { (b, a) };
            (s.and_hms_opt(0, 0, 0)?, e.and_hms_opt(23, 59, 59)?)
        }
        "季度" => {
            let a_dt = add_months(now, sign * (n - 1) * 3)?;
            let a = a_dt.date();
            let b = today;
            let (s, e) = if a <= b { (a, b) } else { (b, a) };
            (s.and_hms_opt(0, 0, 0)?, e.and_hms_opt(23, 59, 59)?)
        }
        _ => return None,
    };
    Some(TimeInfo {
        time_type: "time_span",
        start,
        end,
        definition: "blur",
        ..Default::default()
    })
}

// ───────────────────────── Round 29 — parse_time tail patterns ──────────

/// Pattern #7 — `M月<d1>日、<d2>日、…<dN>日` (enum days).
/// Returns a `time_span` from min(d) to max(d) of the specified month.
fn try_enum_days(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    static RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"^(?:(\d{2,4})\s*年\s*)?(\d{1,2})\s*月\s*((?:\d{1,2}[日号][、,，])+\d{1,2}[日号])$",
        )
        .unwrap()
    });
    let caps = RE.captures(text)?;
    let year = caps
        .get(1)
        .and_then(|m| parse_year(m.as_str()))
        .unwrap_or(now.year());
    let month: u32 = caps.get(2)?.as_str().parse().ok()?;
    let days_str = caps.get(3)?.as_str();
    let mut days: Vec<u32> = Vec::new();
    for part in days_str.split([',', '、', '，']) {
        let p = part.trim_end_matches(['日', '号']).trim();
        if let Ok(d) = p.parse::<u32>() {
            days.push(d);
        }
    }
    if days.is_empty() {
        return None;
    }
    let dmin = *days.iter().min().unwrap();
    let dmax = *days.iter().max().unwrap();
    let (s1, _) = date_range(year, Some(month), Some(dmin))?;
    let (_, e2) = date_range(year, Some(month), Some(dmax))?;
    Some(TimeInfo {
        time_type: "time_span",
        start: s1,
        end: e2,
        definition: "accurate",
        ..Default::default()
    })
}

/// Pattern #10 — `今年/明年/去年 (的)? (前|后|头|最后) N 个?(月|季度)`.
fn try_limit_year_span_month(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let (off, body) = limit_year_prefix(text)?;
    let body = body.strip_prefix('的').unwrap_or(body);
    let year = now.year() + off;
    // Rebuild "YYYY年" + body so try_year_span_month_or_quarter can parse.
    let synth = format!("{}年{}", year, body);
    try_year_span_month_or_quarter(&synth, now)
}

/// Pattern #13/#14 — `YYYY年第N季度(初|中|末|底)` — season boundary.
fn try_year_solar_season_boundary(text: &str) -> Option<TimeInfo> {
    // Case 1: explicit Arabic year.
    static RE_ARAB: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"^(\d{2,4})\s*年\s*(?:第)?\s*([1-4一二三四])\s*季度\s*(初|中|末|底)$").unwrap()
    });
    if let Some(caps) = RE_ARAB.captures(text) {
        let year = parse_year(caps.get(1)?.as_str())?;
        let n_str = caps.get(2)?.as_str();
        let q: u32 = n_str.parse::<u32>().ok().or_else(|| cn_int(n_str))?;
        let pos = caps.get(3)?.as_str();
        return build_quarter_boundary(year, q, pos);
    }
    // Case 2: Chinese year.
    static RE_CN: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"^([零〇一二三四五六七八九十两]+)\s*年\s*(?:第)?\s*([1-4一二三四])\s*季度\s*(初|中|末|底)$",
        )
        .unwrap()
    });
    if let Some(caps) = RE_CN.captures(text) {
        let year = parse_chinese_year_digits(caps.get(1)?.as_str())?;
        let n_str = caps.get(2)?.as_str();
        let q: u32 = n_str.parse::<u32>().ok().or_else(|| cn_int(n_str))?;
        let pos = caps.get(3)?.as_str();
        return build_quarter_boundary(year, q, pos);
    }
    None
}

fn build_quarter_boundary(year: i32, q: u32, pos: &str) -> Option<TimeInfo> {
    if !(1..=4).contains(&q) {
        return None;
    }
    let start_month = (q - 1) * 3 + 1;
    let month = match pos {
        "初" => start_month,
        "中" => start_month + 1,
        "末" | "底" => start_month + 2,
        _ => return None,
    };
    let first = NaiveDate::from_ymd_opt(year, month, 1)?;
    let next = if month == 12 {
        NaiveDate::from_ymd_opt(year + 1, 1, 1)?
    } else {
        NaiveDate::from_ymd_opt(year, month + 1, 1)?
    };
    let last = next.pred_opt()?;
    Some(TimeInfo {
        time_type: "time_span",
        start: first.and_hms_opt(0, 0, 0)?,
        end: last.and_hms_opt(23, 59, 59)?,
        definition: "blur",
        ..Default::default()
    })
}

/// Relative-year + quarter + boundary (`去年四季度初`, `明年一季度末`).
fn try_relative_year_quarter_boundary(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let (offset, rest) = limit_year_prefix(text)?;
    let rest = rest.trim_start_matches('的');
    static RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"^(?:第)?\s*([1-4一二三四])\s*季度\s*(初|中|末|底)$").unwrap());
    let caps = RE.captures(rest)?;
    let n_str = caps.get(1)?.as_str();
    let q: u32 = n_str.parse::<u32>().ok().or_else(|| cn_int(n_str))?;
    let pos = caps.get(2)?.as_str();
    build_quarter_boundary(now.year() + offset, q, pos)
}

/// Limit-quarter + boundary (`上季度末`, `本季度初`, `下季度中`).
fn try_limit_quarter_boundary(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    const Q_PREF: &[(&str, i32)] = &[
        ("本季度", 0),
        ("这季度", 0),
        ("当季度", 0),
        ("上季度", -1),
        ("上个季度", -1),
        ("上一季度", -1),
        ("下季度", 1),
        ("下个季度", 1),
        ("下一季度", 1),
    ];
    let mut best: Option<(i32, &str)> = None;
    for (pref, off) in Q_PREF {
        if let Some(rest) = text.strip_prefix(*pref) {
            match best {
                Some((_, r)) if r.len() < rest.len() => {}
                _ => best = Some((*off, rest)),
            }
        }
    }
    let (offset, rest) = best?;
    let pos = match rest.trim() {
        "初" | "中" | "末" | "底" => rest.trim(),
        _ => return None,
    };
    let cur_q = ((now.month() - 1) / 3) as i32 + 1;
    let mut q = cur_q + offset;
    let mut y = now.year();
    while q < 1 {
        q += 4;
        y -= 1;
    }
    while q > 4 {
        q -= 4;
        y += 1;
    }
    build_quarter_boundary(y, q as u32, pos)
}

/// Pattern #17 — `YYYY年暑假/寒假/春假` (school breaks). Also accepts
/// relative-year prefixes (`去年暑假`, `明年寒假`).
fn try_school_break(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    // First try: relative-year prefix strip.
    let (year, rest_after_year) = if let Some((off, rest)) = limit_year_prefix(text) {
        (now.year() + off, rest.trim_start_matches('的'))
    } else {
        // Fallback to the old (optional Arabic year) regex.
        static RE: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"^(?:(\d{2,4})\s*年\s*)?(暑假|寒假|春假|秋假)$").unwrap());
        let caps = RE.captures(text)?;
        let year = caps
            .get(1)
            .and_then(|m| parse_year(m.as_str()))
            .unwrap_or(now.year());
        let kind = caps.get(2)?.as_str();
        return school_break_range(year, kind);
    };
    // relative-year path: `rest_after_year` should now be just the break name.
    let kind = rest_after_year.trim();
    if !matches!(kind, "暑假" | "寒假" | "春假" | "秋假") {
        return None;
    }
    school_break_range(year, kind)
}

fn school_break_range(year: i32, kind: &str) -> Option<TimeInfo> {
    let (from_y, from_m, to_y, to_m) = match kind {
        "暑假" => (year, 7u32, year, 8u32),
        "寒假" => (year, 1, year, 2),
        "春假" => (year, 4, year, 4),
        "秋假" => (year, 10, year, 10),
        _ => return None,
    };
    let first = NaiveDate::from_ymd_opt(from_y, from_m, 1)?;
    let next = if to_m == 12 {
        NaiveDate::from_ymd_opt(to_y + 1, 1, 1)?
    } else {
        NaiveDate::from_ymd_opt(to_y, to_m + 1, 1)?
    };
    let last = next.pred_opt()?;
    Some(TimeInfo {
        time_type: "time_span",
        start: first.and_hms_opt(0, 0, 0)?,
        end: last.and_hms_opt(23, 59, 59)?,
        definition: "blur",
        ..Default::default()
    })
}

/// Pattern #48 — `今年/明年/去年 第N周` — limit year + week.
fn try_limit_year_week(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let (off, body) = limit_year_prefix(text)?;
    let year = now.year() + off;
    let synth = format!("{}年{}", year, body);
    try_year_week(&synth)
}

/// Pattern #54 — `第N年` — year ordinal. `第一年` ≈ now's year.
fn try_year_ordinal(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    static RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"^第\s*(\d+|[一二两三四五六七八九十]+)\s*年$").unwrap());
    let caps = RE.captures(text)?;
    let n_str = caps.get(1)?.as_str();
    let n: i32 = n_str
        .parse::<i32>()
        .ok()
        .or_else(|| cn_int(n_str).map(|v| v as i32))?;
    // Convention: 第一年 == current year. 第N年 (N>1) == year (now+N-1).
    let year = now.year() + n - 1;
    let first = NaiveDate::from_ymd_opt(year, 1, 1)?;
    let last = NaiveDate::from_ymd_opt(year, 12, 31)?;
    Some(TimeInfo {
        time_type: "time_span",
        start: first.and_hms_opt(0, 0, 0)?,
        end: last.and_hms_opt(23, 59, 59)?,
        definition: "accurate",
        ..Default::default()
    })
}

/// Pattern #96 — `一年四季` — special phrase meaning "throughout the year".
/// Returns a `time_delta` `{year: 1}` tagged blur.
fn try_yinian_siji(text: &str) -> Option<TimeInfo> {
    if text == "一年四季" || text == "全年四季" {
        let mut d = TimeDelta::default();
        d.year = Some(DeltaValue::Single(1.0));
        return Some(TimeInfo {
            time_type: "time_delta",
            definition: "blur",
            delta: Some(d),
            ..Default::default()
        });
    }
    None
}

/// Pattern #98 — `每周工作日(<clock>)?` — weekday-filtered recurrence.
/// Returns time_period with delta {day: 1} and point_string "周工作日(<clock>)".
fn try_recurring_weekday_filtered(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let body = text
        .strip_prefix("每周工作日")
        .or_else(|| text.strip_prefix("每工作日"))?;
    // Find next weekday (Mon-Fri) from now.
    let mut date = now.date() + Duration::days(1);
    while date.weekday().num_days_from_monday() >= 5 {
        date += Duration::days(1);
    }
    let tail = body.trim();
    let (start, end, point_str) = if tail.is_empty() {
        (
            date.and_hms_opt(0, 0, 0)?,
            date.and_hms_opt(23, 59, 59)?,
            "工作日".to_string(),
        )
    } else if let Some((lc, rc)) = parse_clock_range(tail) {
        (
            date.and_time(lc),
            date.and_time(rc),
            format!("工作日{}", tail),
        )
    } else {
        let c = parse_clock(tail)?;
        let dt = date.and_time(c);
        (dt, dt, format!("工作日{}", tail))
    };
    let mut delta = TimeDelta::default();
    delta.day = Some(DeltaValue::Single(1.0));
    Some(TimeInfo {
        time_type: "time_period",
        start,
        end,
        definition: "accurate",
        delta: Some(delta.clone()),
        period: Some(TimePeriodInfo {
            delta,
            point_time: vec![(start, end)],
            point_string: point_str,
        }),
    })
}

/// Pattern #70 — `N<unit1>的M<unit2>(后|前)` — nested delta.
/// E.g. `一个季度的十五分后` = now + 3 months + 15 minutes.
fn try_nested_delta_point(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let idx = text.find('的')?;
    let left = &text[..idx];
    let right = &text[idx + '的'.len_utf8()..];
    if left.is_empty() || right.is_empty() {
        return None;
    }
    // Left must be a pure delta shape (no 前/后). Use try_pure_delta to get
    // the outer TimeDelta.
    let left_info = try_pure_delta(left)?;
    let outer = left_info.delta.as_ref()?;
    // Right must parse as a delta_point (e.g. `十五分后` / `5分钟前`).
    let right_info = try_time_delta(right, now)?;
    // Accept only time_point output (flat offset).
    if right_info.time_type != "time_point" {
        return None;
    }
    // Apply outer on top of right's endpoint.
    let combined = apply_time_delta(right_info.start, outer)?;
    Some(TimeInfo {
        time_type: "time_point",
        start: combined,
        end: combined,
        definition: "accurate",
        ..Default::default()
    })
}

// ───────────────────────── Round 20 — HMS + period + span bounds ─────────

/// Map of Chinese blur-hour qualifiers to their time window `(start_hour,
/// end_hour)`. Matches Python's `blur_time_info_map`.
const BLUR_HOUR_MAP: &[(&str, u32, u32)] = &[
    ("清晨", 5, 7),
    ("清早", 5, 8),
    ("早上", 6, 9),
    ("早晨", 6, 9),
    ("一早", 6, 9),
    ("一大早", 6, 9),
    ("黎明", 4, 6),
    ("白天", 6, 18),
    ("上午", 7, 11),
    ("中午", 12, 13),
    ("午后", 13, 14),
    ("下午", 13, 17),
    ("傍晚", 17, 18),
    ("晚上", 18, 23),
    ("晚间", 20, 23),
    ("夜间", 20, 23),
    ("夜里", 20, 23),
    ("深夜", 23, 23),
    ("上半夜", 0, 2),
    ("前半夜", 0, 2),
    ("下半夜", 2, 4),
    ("后半夜", 2, 4),
    ("半夜", 0, 4),
    ("凌晨", 0, 4),
    ("午夜", 0, 0),
];

/// Pattern #79-#84 — bare blur-hour phrase (今天/明天 + blur-hour).
fn try_blur_hour_phrase(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    // Optional relative-day prefix: 今天/明天/后天/昨天/前天 (and variants).
    const DAY_PREFIXES: &[(&str, i64)] = &[
        ("大前天", -3),
        ("大后天", 3),
        ("前天", -2),
        ("后天", 2),
        ("昨天", -1),
        ("昨日", -1),
        ("明天", 1),
        ("明日", 1),
        ("今天", 0),
        ("今日", 0),
    ];
    let mut day = now.date();
    let mut rest = text;
    for (kw, off) in DAY_PREFIXES {
        if let Some(r) = text.strip_prefix(*kw) {
            day += Duration::days(*off);
            rest = r;
            break;
        }
    }

    // The remainder must equal a known blur-hour exactly.
    for (name, s_hour, e_hour) in BLUR_HOUR_MAP {
        if rest == *name {
            let start = day.and_hms_opt(*s_hour, 0, 0)?;
            let end = day.and_hms_opt(*e_hour, 59, 59)?;
            return Some(TimeInfo {
                time_type: "time_span",
                start,
                end,
                definition: "blur",
                ..Default::default()
            });
        }
    }
    None
}

/// Pattern #85 — `约9点` / `大概下午3点` / `大约上午10点半`. Returns the
/// underlying clock result tagged `"blur"`.
fn try_approx_clock(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    const PREFIXES: &[&str] = &["大约", "大概", "约", "差不多"];
    let mut body: Option<&str> = None;
    for p in PREFIXES {
        if let Some(rest) = text.strip_prefix(*p) {
            body = Some(rest);
            break;
        }
    }
    let body = body?;
    let mut t = parse_time_with_ref(body, now)?;
    t.definition = "blur";
    Some(t)
}

/// Pattern #87 — `前两个小时` / `未来三分钟` / `前一分钟` (HMS super-blur).
fn try_super_blur_hms(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    static RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"^(前|近|过去|未来|这)\s*(\d+|[一二两三四五六七八九十]+)\s*(?:个\s*)?(小时|分钟|秒)$",
        )
        .unwrap()
    });
    let caps = RE.captures(text)?;
    let dir = caps.get(1)?.as_str();
    let n_str = caps.get(2)?.as_str();
    let n: i64 = n_str
        .parse::<i64>()
        .ok()
        .or_else(|| cn_int(n_str).map(|v| v as i64))?;
    let unit = caps.get(3)?.as_str();
    let seconds: i64 = match unit {
        "小时" => n * 3600,
        "分钟" => n * 60,
        "秒" => n,
        _ => return None,
    };
    let sign: i64 = match dir {
        "前" | "近" | "过去" | "这" => -1,
        "未来" => 1,
        _ => return None,
    };
    let endpoint = now + Duration::seconds(seconds * sign);
    let (start, end) = if sign < 0 {
        (endpoint, now)
    } else {
        (now, endpoint)
    };
    Some(TimeInfo {
        time_type: "time_span",
        start,
        end,
        definition: "blur",
        ..Default::default()
    })
}

/// Pattern #107, #108 — open-ended span: `X之后` / `X以后` / `X之前` /
/// `X以前` where X is a concrete date/datetime (not a delta). Emits a
/// half-infinite span by using sentinel bounds (year 9999 / year 1).
fn try_open_ended_span(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    // Suffix first; longer wins. Python also treats a bare `前` after
    // a year-boundary (`年底前`, `明年底前`) as an open-ended "before"
    // span, so accept bare `前` when the body already ends in a blur
    // token like `底/初/末/中`.
    let (is_after, body) = if let Some(b) = text
        .strip_suffix("之后")
        .or_else(|| text.strip_suffix("以后"))
    {
        (true, b)
    } else if let Some(b) = text
        .strip_suffix("之前")
        .or_else(|| text.strip_suffix("以前"))
    {
        (false, b)
    } else if let Some(b) = text.strip_suffix('前') {
        // Bare `前` after year-boundary (`年初`, `年底` etc.), month-
        // boundary (`8月初`, `本月底`), OR after a concrete
        // `YYYY年` / `YYYY年M月` that resolves to a future point.
        // Reject pure duration forms (`三年前` = 3 years ago).
        let last = b.chars().last()?;
        let is_boundary_suffix = matches!(last, '初' | '末' | '底' | '中' | '头' | '尾');
        let is_concrete_year = (last == '年' || last == '月')
            && parse_time_with_ref(b, now)
                .map(|t| t.start >= now)
                .unwrap_or(false);
        if !(is_boundary_suffix || is_concrete_year) {
            return None;
        }
        (false, b)
    } else {
        return None;
    };
    // Guard: reject delta-form bodies (`3月之后` / `5天之前`). A body is
    // delta-shaped if the last char is a unit AND the body does not also
    // contain a year token (`YYYY年` or `今年/明年/去年`). Year-prefixed
    // dates like `2024年3月之后` or `春节之前` are concrete and go through.
    let body_trim = body.trim();
    let contains_year_prefix = body_trim.contains('年')
        || body_trim.starts_with("今年")
        || body_trim.starts_with("明年")
        || body_trim.starts_with("去年")
        || body_trim.starts_with("后年")
        || body_trim.starts_with("前年");
    if !contains_year_prefix {
        if let Some(last) = body_trim.chars().last() {
            if matches!(last, '月' | '日' | '天' | '周' | '时' | '分' | '秒') {
                let before = &body_trim[..body_trim.len() - last.len_utf8()];
                if !before.is_empty()
                    && before
                        .chars()
                        .last()
                        .map(|c| {
                            c.is_ascii_digit()
                                || matches!(
                                    c,
                                    '一' | '二'
                                        | '两'
                                        | '三'
                                        | '四'
                                        | '五'
                                        | '六'
                                        | '七'
                                        | '八'
                                        | '九'
                                        | '十'
                                        | '百'
                                        | '千'
                                        | '万'
                                        | '半'
                                )
                        })
                        .unwrap_or(false)
                {
                    return None;
                }
            }
        }
    }
    // Parse the body as a standard concrete time.
    let inner = parse_time_with_ref(body, now)?;
    // Only accept time_point or time_span inner.
    if !(inner.time_type == "time_point" || inner.time_type == "time_span") {
        return None;
    }
    // Python convention: use `now` as the non-fixed endpoint when the
    // open side is reachable from now (i.e. `X之前` with future X →
    // [now, X]; `X之后` with past X → [X, now]). Fall back to a far
    // sentinel only when the gap crosses all of history / future.
    let sentinel_far = NaiveDate::from_ymd_opt(9999, 12, 31)?.and_hms_opt(23, 59, 59)?;
    let sentinel_beg = NaiveDate::from_ymd_opt(1, 1, 1)?.and_hms_opt(0, 0, 0)?;
    let (start, end) = if is_after {
        // `X之后` — from start-of-X to +inf (or to `now` if X is past).
        if inner.end <= now {
            (inner.end, now)
        } else {
            (inner.end, sentinel_far)
        }
    } else if inner.end >= now {
        // `X之前` with future X → [now, end-of-X].
        (now, inner.end)
    } else {
        // `X之前` with past X → [-inf, end-of-X].
        (sentinel_beg, inner.end)
    };
    // Definition: `accurate` when both endpoints are known real dates
    // (i.e. Python's case where one endpoint is `now` and the other is
    // concrete), else `blur` (one is sentinel).
    let defn = if start == sentinel_beg || end == sentinel_far {
        "blur"
    } else {
        "accurate"
    };
    Some(TimeInfo {
        time_type: "time_span",
        start,
        end,
        definition: defn,
        ..Default::default()
    })
}

// ───────────────────────── recurring extensions (Round 20) ────────────────

/// Pattern #101, #102 — `每小时` / `每个小时` / `每N分钟` / `每N小时` /
/// `每N秒` → time_period cadence.
fn try_recurring_hms(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let body = text.strip_prefix('每')?.trim_start();
    static RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"^(?:个)?\s*(\d*|[一二两三四五六七八九十]*)\s*(?:个)?\s*(小时|分钟|秒)$")
            .unwrap()
    });
    let caps = RE.captures(body)?;
    let n_str = caps.get(1)?.as_str();
    let n: f64 = if n_str.is_empty() {
        1.0
    } else {
        n_str
            .parse::<f64>()
            .ok()
            .or_else(|| parse_chinese_number(n_str).map(|v| v as f64))?
    };
    let unit = caps.get(2)?.as_str();
    let mut delta = TimeDelta::default();
    match unit {
        "小时" => delta.hour = Some(DeltaValue::Single(n)),
        "分钟" => delta.minute = Some(DeltaValue::Single(n)),
        "秒" => delta.second = Some(DeltaValue::Single(n)),
        _ => return None,
    }
    let seconds: i64 = match unit {
        "小时" => (n * 3600.0) as i64,
        "分钟" => (n * 60.0) as i64,
        "秒" => n as i64,
        _ => 0,
    };
    let next = now + Duration::seconds(seconds);
    Some(TimeInfo {
        time_type: "time_period",
        start: next,
        end: next,
        definition: "accurate",
        delta: Some(delta.clone()),
        period: Some(TimePeriodInfo {
            delta,
            point_time: vec![(next, next)],
            point_string: String::new(),
        }),
    })
}

/// Pattern #103 — `每隔N天` / `每隔一天` → cadence N+1 days.
fn try_recurring_gap(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let body = text.strip_prefix("每隔")?.trim_start();
    static RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"^(\d+|[一二两三四五六七八九十]+)\s*(?:个)?\s*(天|日|周|星期|月|年|小时|分钟|秒)$",
        )
        .unwrap()
    });
    let caps = RE.captures(body)?;
    let n_str = caps.get(1)?.as_str();
    let n: f64 = n_str
        .parse::<f64>()
        .ok()
        .or_else(|| parse_chinese_number(n_str).map(|v| v as f64))?;
    let unit = caps.get(2)?.as_str();
    // "每隔 N" means gap of N, so cadence is N+1.
    let cadence = n + 1.0;
    let mut delta = TimeDelta::default();
    let (next_date, _) = match unit {
        "天" | "日" => {
            delta.day = Some(DeltaValue::Single(cadence));
            let nd = now.date() + Duration::days(cadence as i64);
            (nd, "day")
        }
        "周" | "星期" => {
            delta.day = Some(DeltaValue::Single(cadence * 7.0));
            let nd = now.date() + Duration::days((cadence * 7.0) as i64);
            (nd, "week")
        }
        "月" => {
            delta.month = Some(DeltaValue::Single(cadence));
            let nd_dt = add_months(now, cadence as i32)?;
            (nd_dt.date(), "month")
        }
        "年" => {
            delta.year = Some(DeltaValue::Single(cadence));
            let nd_dt = add_months(now, (cadence * 12.0) as i32)?;
            (nd_dt.date(), "year")
        }
        "小时" => {
            delta.hour = Some(DeltaValue::Single(cadence));
            let nd = (now + Duration::seconds((cadence * 3600.0) as i64)).date();
            (nd, "hour")
        }
        "分钟" => {
            delta.minute = Some(DeltaValue::Single(cadence));
            let nd = (now + Duration::seconds((cadence * 60.0) as i64)).date();
            (nd, "minute")
        }
        "秒" => {
            delta.second = Some(DeltaValue::Single(cadence));
            let nd = (now + Duration::seconds(cadence as i64)).date();
            (nd, "second")
        }
        _ => return None,
    };
    let start = next_date.and_hms_opt(0, 0, 0)?;
    let end = next_date.and_hms_opt(23, 59, 59)?;
    Some(TimeInfo {
        time_type: "time_period",
        start,
        end,
        definition: "accurate",
        delta: Some(delta.clone()),
        period: Some(TimePeriodInfo {
            delta,
            point_time: vec![(start, end)],
            point_string: String::new(),
        }),
    })
}

/// Pattern #100 — `每年 + <festival>` — yearly festival recurrence.
fn try_recurring_yearly_festival(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let body = text.strip_prefix("每年")?.trim_start();
    if body.is_empty() {
        return None;
    }
    // Delegate to festival parser with current year.
    let synthesized = format!("{}年{}", now.year(), body);
    let inner = try_fixed_holiday(&synthesized, now)?;
    let mut delta = TimeDelta::default();
    delta.year = Some(DeltaValue::Single(1.0));
    Some(TimeInfo {
        time_type: "time_period",
        start: inner.start,
        end: inner.end,
        definition: "accurate",
        delta: Some(delta.clone()),
        period: Some(TimePeriodInfo {
            delta,
            point_time: vec![(inner.start, inner.end)],
            point_string: body.to_string(),
        }),
    })
}

// ───────────────────────── Round 19 — delta patterns ─────────────────────

/// Shared unit table for bare delta parsing. The closure sets one of the
/// `TimeDelta` fields with the given `DeltaValue`. Longer unit names are
/// placed before shorter ones so `个月` wins over `月`, etc.
type DeltaUnitSetter = fn(&mut TimeDelta, DeltaValue);
const DELTA_UNIT_TABLE: &[(&str, DeltaUnitSetter)] = &[
    ("个世纪", |d, v| d.year = Some(scale_value(v, 100.0))),
    ("世纪", |d, v| d.year = Some(scale_value(v, 100.0))),
    ("个年代", |d, v| d.year = Some(scale_value(v, 10.0))),
    ("年代", |d, v| d.year = Some(scale_value(v, 10.0))),
    ("个季度", |d, v| d.month = Some(scale_value(v, 3.0))),
    ("季度", |d, v| d.month = Some(scale_value(v, 3.0))),
    ("个小时", |d, v| d.hour = Some(v)),
    ("小时", |d, v| d.hour = Some(v)),
    ("个月", |d, v| d.month = Some(v)),
    ("分钟", |d, v| d.minute = Some(v)),
    ("星期", |d, v| d.day = Some(scale_value(v, 7.0))),
    ("工作日", |d, v| d.workday = Some(v)),
    ("年", |d, v| d.year = Some(v)),
    ("月", |d, v| d.month = Some(v)),
    ("周", |d, v| d.day = Some(scale_value(v, 7.0))),
    ("天", |d, v| d.day = Some(v)),
    ("日", |d, v| d.day = Some(v)),
    ("时", |d, v| d.hour = Some(v)),
    ("分", |d, v| d.minute = Some(v)),
    ("秒", |d, v| d.second = Some(v)),
];

fn scale_value(v: DeltaValue, factor: f64) -> DeltaValue {
    match v {
        DeltaValue::Single(n) => DeltaValue::Single(n * factor),
        DeltaValue::Range(lo, hi) => DeltaValue::Range(lo * factor, hi * factor),
    }
}

/// Consume a trailing unit from the tail of a delta expression. Returns
/// `(setter, body_before_unit)`.
fn strip_delta_unit(body: &str) -> Option<(DeltaUnitSetter, &str)> {
    for (unit, setter) in DELTA_UNIT_TABLE {
        if let Some(prefix) = body.strip_suffix(*unit) {
            return Some((*setter, prefix));
        }
    }
    None
}

/// Pattern #91/#92/#88 — bare `N<unit>` (no 前/后/之后) → `time_delta`
/// dict. E.g. `3年` / `两个月` / `一万个小时`.
fn try_pure_delta(text: &str) -> Option<TimeInfo> {
    // Reject anything containing separators — those belong to other paths.
    if text.contains(['~', '～', '到', '至']) {
        return None;
    }
    let (setter, body) = strip_delta_unit(text.trim())?;
    let body = body.trim().trim_end_matches('个').trim();
    if body.is_empty() {
        return None;
    }

    // Ranges/blur quantifiers must be checked BEFORE parse_chinese_number
    // (which would collapse 两三 → 3). Longest literals first.
    let v = if body == "大半" {
        DeltaValue::Single(0.7)
    } else if body == "半" {
        DeltaValue::Single(0.5)
    } else if body == "好几" {
        DeltaValue::Range(3.0, 8.0)
    } else if body == "几十" {
        // Python uses [20, 100] so `几十年之后` → now+100y.
        DeltaValue::Range(20.0, 100.0)
    } else if body == "十几" {
        DeltaValue::Range(12.0, 18.0)
    } else if let Some(caps) = {
        // `二十几` / `三十几` / ... → Range(X0, X9). Covers
        // `二十几年前` etc.
        static RE: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"^([一二两三四五六七八九])十几$").unwrap());
        RE.captures(body)
    } {
        let d = cn_digit_1_9(caps.get(1)?.as_str().chars().next()?)?;
        // Python uses [N0, (N+1)0], e.g. `二十几年前` ⇒ 20–30 years back.
        DeltaValue::Range((d as f64) * 10.0, (d as f64 + 1.0) * 10.0)
    } else if let Some(caps) = {
        // 两三 / 三四 / 五六 → range, takes priority over parse_chinese_number.
        static RE: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"^([一二两三四五六七八九])([一二三四五六七八九])$").unwrap());
        RE.captures(body)
    } {
        let lo = cn_digit_1_9(caps.get(1)?.as_str().chars().next()?)?;
        let hi = cn_digit_1_9(caps.get(2)?.as_str().chars().next()?)?;
        DeltaValue::Range(lo as f64, hi as f64)
    } else if body == "几" {
        DeltaValue::Range(3.0, 8.0)
    } else if let Ok(n) = body.parse::<f64>() {
        DeltaValue::Single(n)
    } else if let Some(n) = parse_chinese_number(body) {
        DeltaValue::Single(n as f64)
    } else {
        return None;
    };

    let mut d = TimeDelta::default();
    setter(&mut d, v);
    if d.is_empty() {
        return None;
    }
    let definition = if matches!(
        d.year
            .as_ref()
            .or(d.month.as_ref())
            .or(d.day.as_ref())
            .or(d.hour.as_ref())
            .or(d.minute.as_ref())
            .or(d.second.as_ref())
            .or(d.workday.as_ref()),
        Some(DeltaValue::Range(_, _))
    ) {
        "blur"
    } else {
        "accurate"
    };
    Some(TimeInfo {
        time_type: "time_delta",
        definition,
        delta: Some(d),
        ..Default::default()
    })
}

/// Pattern #58, #59, #60, #61 — `未来N<unit>` / `过去N<unit>` / `过N<unit>`
/// / `再过N<unit>` / `最近N<unit>` as `time_span` from now.
fn try_delta_to_span(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    const PREFIXES: &[(&str, i64)] = &[
        ("未来", 1),
        ("将来", 1),
        ("再过", 1),
        ("过", 1),
        ("过去", -1),
        ("最近", -1),
        ("近", -1),
    ];
    let mut matched: Option<(i64, &str)> = None;
    for (p, sign) in PREFIXES {
        if let Some(rest) = text.strip_prefix(*p) {
            match matched {
                Some((_, r)) if r.len() < rest.len() => {}
                _ => matched = Some((*sign, rest)),
            }
        }
    }
    let (sign, rest) = matched?;
    let rest = rest.trim();
    // rest must be "N<unit>", optionally with 个.
    let (setter, body) = strip_delta_unit(rest)?;
    let body = body.trim_end_matches('个').trim();
    let n: f64 = body
        .parse::<f64>()
        .ok()
        .or_else(|| parse_chinese_number(body).map(|n| n as f64))?;

    // Compute span by rolling a TimeDelta through add_seconds-equivalent.
    let mut tmp = TimeDelta::default();
    setter(&mut tmp, DeltaValue::Single(n * sign as f64));
    let endpoint = apply_time_delta(now, &tmp)?;

    let (start, end) = if sign > 0 {
        (now, endpoint)
    } else {
        (endpoint, now)
    };
    Some(TimeInfo {
        time_type: "time_span",
        start,
        end,
        definition: "blur",
        ..Default::default()
    })
}

/// Pattern #65, #66 — `N<unit>内` / `N<unit>来` → time_span ending at now,
/// starting N<unit> earlier. Longest suffix first so 之内/以内 win over bare 内.
fn try_delta_inner_span(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let body = text
        .strip_suffix("之内")
        .or_else(|| text.strip_suffix("以内"))
        .or_else(|| text.strip_suffix('内'))
        .or_else(|| text.strip_suffix('来'));
    let body = body?;
    let (setter, rest) = strip_delta_unit(body)?;
    let rest = rest.trim_end_matches('个').trim();
    let n: f64 = rest
        .parse::<f64>()
        .ok()
        .or_else(|| parse_chinese_number(rest).map(|n| n as f64))?;
    let mut tmp = TimeDelta::default();
    setter(&mut tmp, DeltaValue::Single(-n));
    let start = apply_time_delta(now, &tmp)?;
    Some(TimeInfo {
        time_type: "time_span",
        start,
        end: now,
        definition: "blur",
        ..Default::default()
    })
}

/// Pattern #64, #67 — sub-day-precision `...之后` / `...之前` / `...以后` /
/// `...以前` → time_span from now (or to now) over the delta window. Only
/// fires for hour/minute/second/工作日 units; day-and-above goes via
/// try_time_delta to preserve the existing day-precision time_point output.
fn try_delta_open_ended_span_subday(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let (sign, body) = if let Some(rest) = text
        .strip_suffix("之后")
        .or_else(|| text.strip_suffix("以后"))
    {
        (1i64, rest)
    } else if let Some(rest) = text
        .strip_suffix("之前")
        .or_else(|| text.strip_suffix("以前"))
    {
        (-1, rest)
    } else {
        return None;
    };
    // Only fire for sub-day units.
    let is_subday = body.ends_with("小时")
        || body.ends_with("分钟")
        || body.ends_with('秒')
        || body.ends_with("个小时");
    if !is_subday {
        return None;
    }
    let (setter, rest) = strip_delta_unit(body)?;
    let rest = rest.trim_end_matches('个').trim();
    let n: f64 = rest
        .parse::<f64>()
        .ok()
        .or_else(|| parse_chinese_number(rest).map(|n| n as f64))?;
    let mut tmp = TimeDelta::default();
    setter(&mut tmp, DeltaValue::Single(n * sign as f64));
    let endpoint = apply_time_delta(now, &tmp)?;
    let (start, end) = if sign > 0 {
        (now, endpoint)
    } else {
        (endpoint, now)
    };
    Some(TimeInfo {
        time_type: "time_span",
        start,
        end,
        definition: "blur",
        ..Default::default()
    })
}

/// Pattern #69 — `N个工作日前` / `N个工作日后`. Returns time_point shifted
/// by N business days (skipping weekends).
fn try_workday_delta_point(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let (sign, body) = if let Some(rest) = text
        .strip_suffix("之后")
        .or_else(|| text.strip_suffix("以后"))
    {
        (1i64, rest)
    } else if let Some(rest) = text
        .strip_suffix("之前")
        .or_else(|| text.strip_suffix("以前"))
    {
        (-1, rest)
    } else if let Some(rest) = text.strip_suffix('后') {
        (1, rest)
    } else if let Some(rest) = text.strip_suffix('前') {
        (-1, rest)
    } else {
        return None;
    };
    let rest = body.strip_suffix("工作日")?.trim_end_matches('个').trim();
    let n: i64 = rest
        .parse::<i64>()
        .ok()
        .or_else(|| parse_chinese_number(rest).map(|v| v as i64))?;

    // Shift by n workdays.
    let mut date = now.date();
    let mut remaining = n * sign;
    let step = if remaining >= 0 { 1i64 } else { -1 };
    while remaining != 0 {
        date += Duration::days(step);
        let wd = date.weekday().num_days_from_monday();
        if wd < 5 {
            remaining -= step;
        }
    }
    let dt = date.and_hms_opt(0, 0, 0)?;
    Some(TimeInfo {
        time_type: "time_point",
        start: dt,
        end: dt,
        definition: "accurate",
        ..Default::default()
    })
}

/// Apply a TimeDelta (Single-valued) to a reference time. Used by the
/// Round 19 span parsers.
fn apply_time_delta(now: NaiveDateTime, d: &TimeDelta) -> Option<NaiveDateTime> {
    let mut result = now;
    if let Some(DeltaValue::Single(n)) = d.year {
        result = add_months(result, (n * 12.0) as i32)?;
    }
    if let Some(DeltaValue::Single(n)) = d.month {
        // Fractional months → day-level approximation.
        let whole = n as i32;
        result = add_months(result, whole)?;
        let frac = n - whole as f64;
        if frac.abs() > 1e-9 {
            result += Duration::seconds((frac * 86400.0 * 30.0) as i64);
        }
    }
    if let Some(DeltaValue::Single(n)) = d.day {
        result += Duration::seconds((n * 86400.0) as i64);
    }
    if let Some(DeltaValue::Single(n)) = d.hour {
        result += Duration::seconds((n * 3600.0) as i64);
    }
    if let Some(DeltaValue::Single(n)) = d.minute {
        result += Duration::seconds((n * 60.0) as i64);
    }
    if let Some(DeltaValue::Single(n)) = d.second {
        result += Duration::seconds(n as i64);
    }
    if let Some(DeltaValue::Single(n)) = d.workday {
        // Rough: 1 workday ≈ 1 day (ignores weekends).
        result += Duration::days(n as i64);
    }
    Some(result)
}

// ───────────────────────── Round 18 — lunar / solar terms / seasons ──────

/// 24 solar terms (节气). Entry: `(name, month, key_20th, key_21st, corrections)`.
/// The "day" of the term in a given year is computed as:
///   `day = floor(key * (year mod 100) - floor((year mod 100 - 1) / 4))`
/// plus per-year corrections, per Python reference implementation.
type SolarTermEntry = (&'static str, u32, f64, f64, &'static [(i32, i32)]);
const SOLAR_TERMS: &[SolarTermEntry] = &[
    ("小寒", 1, 6.11, 5.4055, &[(2019, -1), (1982, 1)]),
    ("大寒", 1, 20.84, 20.12, &[(2082, 1)]),
    ("立春", 2, 4.6295, 3.87, &[]),
    ("雨水", 2, 19.4599, 18.73, &[(2026, -1)]),
    ("惊蛰", 3, 6.3826, 5.63, &[]),
    ("春分", 3, 21.4155, 20.646, &[(2084, 1)]),
    ("清明", 4, 5.59, 4.81, &[]),
    ("谷雨", 4, 20.888, 20.1, &[]),
    ("立夏", 5, 6.318, 5.52, &[(1911, 1)]),
    ("小满", 5, 21.86, 21.04, &[(2008, 1)]),
    ("芒种", 6, 6.5, 5.678, &[(1902, 1)]),
    ("夏至", 6, 22.2, 21.37, &[]),
    ("小暑", 7, 7.928, 7.108, &[(2016, 1), (1925, 1)]),
    ("大暑", 7, 23.65, 22.83, &[(1922, 1)]),
    ("立秋", 8, 8.35, 7.5, &[(2002, 1)]),
    ("处暑", 8, 23.95, 23.13, &[]),
    ("白露", 9, 8.44, 7.646, &[(1927, 1)]),
    ("秋分", 9, 23.822, 23.042, &[]),
    ("寒露", 10, 9.098, 8.318, &[(2088, 0)]),
    ("霜降", 10, 24.218, 23.438, &[(2089, 1)]),
    ("立冬", 11, 8.218, 7.438, &[(2089, 1)]),
    ("小雪", 11, 23.08, 22.36, &[(1978, 0)]),
    ("大雪", 12, 7.9, 7.18, &[(1954, 1)]),
    ("冬至", 12, 22.6, 21.94, &[(2021, -1), (1918, -1)]),
];

fn solar_term_date(term: &str, year: i32) -> Option<(u32, u32)> {
    for &(name, month, key20, key21, corrections) in SOLAR_TERMS {
        if name == term {
            let y_mod = (year % 100) as f64;
            let key = if year < 2000 { key20 } else { key21 };
            // Shouxi formula: day = Y*0.2422 + key - floor((Y-1)/4), then
            // apply per-year corrections for the well-known edge cases.
            let day_f = y_mod * 0.2422 + key - ((y_mod - 1.0) / 4.0).floor();
            let mut day = day_f as i32;
            for (yr, delta) in corrections {
                if year == *yr {
                    day += delta;
                }
            }
            if !(1..=31).contains(&day) {
                return None;
            }
            return Some((month, day as u32));
        }
    }
    None
}

/// Pattern #36, #37 — solar term (optionally year-prefixed or limit-prefixed).
fn try_solar_term(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    // Optional prefix: YYYY年 / 零三年 / 今年/明年/去年.
    let (year, rest) = {
        // ASCII year prefix.
        if let Some(caps) = {
            static RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(\d{2,4})\s*年\s*(.+)$").unwrap());
            RE.captures(text)
        } {
            (
                parse_year(caps.get(1)?.as_str())?,
                caps.get(2)?.as_str().to_string(),
            )
        } else if let Some(caps) = {
            static RE: Lazy<Regex> =
                Lazy::new(|| Regex::new(r"^([零〇一二三四五六七八九十两]+)\s*年\s*(.+)$").unwrap());
            RE.captures(text)
        } {
            (
                parse_chinese_year_digits(caps.get(1)?.as_str())?,
                caps.get(2)?.as_str().to_string(),
            )
        } else if let Some((off, body)) = limit_year_prefix(text) {
            (now.year() + off, body.to_string())
        } else {
            (now.year(), text.to_string())
        }
    };

    for &(name, _, _, _, _) in SOLAR_TERMS {
        if rest == name {
            let (m, d) = solar_term_date(name, year)?;
            let (start, end) = date_range(year, Some(m), Some(d))?;
            return Some(TimeInfo {
                time_type: "time_point",
                start,
                end,
                definition: "accurate",
                ..Default::default()
            });
        }
    }
    None
}

/// Strip a `今年/明年/去年/后年/前年` limit-year prefix. Returns
/// `(year_offset, remainder)`.
fn limit_year_prefix(text: &str) -> Option<(i32, &str)> {
    const TABLE: &[(&str, i32)] = &[
        ("前年", -2),
        ("大前年", -3),
        ("去年", -1),
        ("今年", 0),
        ("明年", 1),
        ("后年", 2),
        ("大后年", 3),
    ];
    // Longest prefix wins.
    let mut best: Option<(i32, &str)> = None;
    for (kw, off) in TABLE {
        if let Some(rest) = text.strip_prefix(*kw) {
            match best {
                Some((_, r)) if r.len() < rest.len() => {}
                _ => best = Some((*off, rest)),
            }
        }
    }
    best
}

/// Season → (first_month, last_month) mapping. Python uses solar-term
/// boundaries; for readability we use calendar-quarter approximation which
/// matches Python's `normalize_year_lunar_season` output on the month-start
/// and month-end boundaries.
fn season_months(name: &str) -> Option<(u32, u32)> {
    match name {
        "春" | "春天" | "春季" => Some((3, 5)),
        "夏" | "夏天" | "夏季" => Some((6, 8)),
        "秋" | "秋天" | "秋季" => Some((9, 11)),
        "冬" | "冬天" | "冬季" => Some((12, 12)), // Dec only; Jan+Feb handled via span
        _ => None,
    }
}

/// Pattern #38, #39 — seasons with optional year or limit-year prefix.
fn try_season(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let (year, rest) = {
        if let Some(caps) = {
            static RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(\d{2,4})\s*年\s*(.+)$").unwrap());
            RE.captures(text)
        } {
            (
                parse_year(caps.get(1)?.as_str())?,
                caps.get(2)?.as_str().to_string(),
            )
        } else if let Some((off, body)) = limit_year_prefix(text) {
            (now.year() + off, body.to_string())
        } else {
            (now.year(), text.to_string())
        }
    };
    let (m_start, m_end) = season_months(&rest)?;
    let first = NaiveDate::from_ymd_opt(year, m_start, 1)?;
    // 冬 wraps to next year's Feb 28/29 for "冬天".
    let (end_year, end_month) = if rest == "冬" || rest == "冬天" || rest == "冬季" {
        (year + 1, 2u32)
    } else {
        (year, m_end)
    };
    let next = if end_month == 12 {
        NaiveDate::from_ymd_opt(end_year + 1, 1, 1)?
    } else {
        NaiveDate::from_ymd_opt(end_year, end_month + 1, 1)?
    };
    let last = next.pred_opt()?;
    Some(TimeInfo {
        time_type: "time_span",
        start: first.and_hms_opt(0, 0, 0)?,
        end: last.and_hms_opt(23, 59, 59)?,
        definition: "accurate",
        ..Default::default()
    })
}

// ───────────────────────── Lunar calendar parsing ──────────────────────────

/// Parse a bare `N月` string (e.g. `"4月"`) to month number. Returns
/// `None` if the input isn't exactly `<digits>月`.
fn parse_bare_month(s: &str) -> Option<u32> {
    let s = s.trim();
    let rest = s.strip_suffix('月')?;
    let m: u32 = rest.trim().parse().ok()?;
    if (1..=12).contains(&m) {
        Some(m)
    } else {
        None
    }
}

/// Lunar month name → month number (1-12). `闰` prefix flags a leap month.
/// Accepts ASCII digit form (`9月`) too, to parity with Python.
fn parse_lunar_month(s: &str) -> Option<(u32, bool, &str)> {
    let (is_leap, body) = if let Some(rest) = s.strip_prefix('闰') {
        (true, rest)
    } else {
        (false, s)
    };

    // ASCII digit month: "N月".
    let bytes = body.as_bytes();
    let mut end = 0usize;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end > 0 {
        if let Some(rest) = body[end..].strip_prefix('月') {
            let m: u32 = body[..end].parse().ok()?;
            if (1..=12).contains(&m) {
                return Some((m, is_leap, rest));
            }
        }
    }

    const MONTHS: &[(&str, u32)] = &[
        ("正月", 1),
        ("冬月", 11),
        ("腊月", 12),
        ("梅月", 1),
        ("一月", 1),
        ("二月", 2),
        ("三月", 3),
        ("四月", 4),
        ("五月", 5),
        ("六月", 6),
        ("七月", 7),
        ("八月", 8),
        ("九月", 9),
        ("十月", 10),
        ("十一月", 11),
        ("十二月", 12),
    ];
    // Longest wins (十一月 vs 一月).
    let mut best: Option<(u32, usize)> = None;
    for (name, n) in MONTHS {
        if body.starts_with(name) {
            let len = name.len();
            match best {
                Some((_, l)) if l >= len => {}
                _ => best = Some((*n, len)),
            }
        }
    }
    let (m, consumed) = best?;
    Some((m, is_leap, &body[consumed..]))
}

/// Lunar day name → day number (1-30). Handles: 初一..初十 / 十一..十九 /
/// 二十 / 二十一..二十九 / 三十 / plain Chinese numeral day.
fn parse_lunar_day(s: &str) -> Option<(u32, &str)> {
    // Try ASCII digit first.
    let bytes = s.as_bytes();
    let mut end = 0usize;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end > 0 {
        let d: u32 = s[..end].parse().ok()?;
        if (1..=30).contains(&d) {
            return Some((d, &s[end..]));
        }
    }
    // 初一 .. 初十. Also accept `初` + Arabic digit (e.g. `初8`).
    if let Some(rest) = s.strip_prefix('初') {
        // 初十 first — longer prefix wins vs ambiguous single-char match.
        if let Some(r) = rest.strip_prefix('十') {
            return Some((10, r));
        }
        // Arabic digit after 初.
        let rest_bytes = rest.as_bytes();
        if !rest_bytes.is_empty() && rest_bytes[0].is_ascii_digit() {
            let mut end = 0;
            while end < rest_bytes.len() && rest_bytes[end].is_ascii_digit() {
                end += 1;
            }
            if let Ok(d) = rest[..end].parse::<u32>() {
                if (1..=9).contains(&d) {
                    return Some((d, &rest[end..]));
                }
            }
        }
        if let Some((n, rest2)) = next_single_cn_digit(rest) {
            if let Some(r2) = rest2 {
                if (1..=9).contains(&n) {
                    return Some((n, r2));
                }
            }
        }
    }
    // 三十 / 二十 / 二十N / 十一..十九 / 十.
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    // 三十 prefix.
    if chars.len() >= 2 && chars[0].1 == '三' && chars[1].1 == '十' {
        let end_idx = if chars.len() > 2 { chars[2].0 } else { s.len() };
        return Some((30, &s[end_idx..]));
    }
    // 二十 or 二十N.
    if chars.len() >= 2 && chars[0].1 == '二' && chars[1].1 == '十' {
        if chars.len() >= 3 {
            if let Some(u) = cn_digit_1_9(chars[2].1) {
                let end_idx = if chars.len() > 3 { chars[3].0 } else { s.len() };
                return Some((20 + u, &s[end_idx..]));
            }
        }
        let end_idx = if chars.len() > 2 { chars[2].0 } else { s.len() };
        return Some((20, &s[end_idx..]));
    }
    // 十一..十九 or 十.
    if !chars.is_empty() && chars[0].1 == '十' {
        if chars.len() >= 2 {
            if let Some(u) = cn_digit_1_9(chars[1].1) {
                let end_idx = if chars.len() > 2 { chars[2].0 } else { s.len() };
                return Some((10 + u, &s[end_idx..]));
            }
        }
        let end_idx = if chars.len() > 1 { chars[1].0 } else { s.len() };
        return Some((10, &s[end_idx..]));
    }
    // 廿一..廿九 or 廿 (= 20+N); 卅一..卅九 or 卅 (= 30+N); 念 ≡ 廿,
    // 丗 ≡ 卅. Used for lunar day short-hand (`廿二` = 22).
    if !chars.is_empty() && matches!(chars[0].1, '廿' | '念') {
        if chars.len() >= 2 {
            if let Some(u) = cn_digit_1_9(chars[1].1) {
                let end_idx = if chars.len() > 2 { chars[2].0 } else { s.len() };
                return Some((20 + u, &s[end_idx..]));
            }
        }
        let end_idx = if chars.len() > 1 { chars[1].0 } else { s.len() };
        return Some((20, &s[end_idx..]));
    }
    if !chars.is_empty() && matches!(chars[0].1, '卅' | '丗') {
        if chars.len() >= 2 {
            if let Some(u) = cn_digit_1_9(chars[1].1) {
                let end_idx = if chars.len() > 2 { chars[2].0 } else { s.len() };
                return Some((30 + u, &s[end_idx..]));
            }
        }
        let end_idx = if chars.len() > 1 { chars[1].0 } else { s.len() };
        return Some((30, &s[end_idx..]));
    }
    None
}

fn cn_digit_1_9(c: char) -> Option<u32> {
    match c {
        '一' => Some(1),
        '二' | '两' => Some(2),
        '三' => Some(3),
        '四' => Some(4),
        '五' => Some(5),
        '六' => Some(6),
        '七' => Some(7),
        '八' => Some(8),
        '九' => Some(9),
        _ => None,
    }
}

fn next_single_cn_digit(s: &str) -> Option<(u32, Option<&str>)> {
    let mut it = s.char_indices();
    let (_, c) = it.next()?;
    let n = cn_digit_1_9(c)?;
    let rest = it.next().map(|(i, _)| &s[i..]).or(Some(""));
    Some((n, rest))
}

/// Pattern #24-#28 — lunar date with optional 农历 marker & year prefix.
fn try_lunar_date(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    use crate::gadget::lunar_solar::lunar_to_solar;

    // Step 1: peel optional year. Three shapes:
    //   "YYYY年农历MMDD" / "农历YYYY年MMDD"
    //   "今年/明年/去年 农历MMDD"
    //   "农历MMDD" (current lunar year)
    //   "腊月初八" (no 农历 marker; standalone lunar month implies lunar)
    let (year, body) = {
        // `农历` and `阴历` are synonyms; also accept the prefix mid-sentence
        // (e.g. `阴历二〇二一年六月`).
        let stripped_prefix = text
            .strip_prefix("农历")
            .or_else(|| text.strip_prefix("阴历"))
            .unwrap_or(text);
        // Year?
        if let Some(caps) = {
            static RE: Lazy<Regex> =
                Lazy::new(|| Regex::new(r"^(\d{2,4})\s*年\s*(?:(?:农历|阴历)\s*)?(.+)$").unwrap());
            RE.captures(stripped_prefix)
        } {
            (
                parse_year(caps.get(1)?.as_str())?,
                caps.get(2)?.as_str().to_string(),
            )
        } else if let Some(caps) = {
            static RE: Lazy<Regex> = Lazy::new(|| {
                Regex::new(r"^([零〇一二三四五六七八九十两]+)\s*年\s*(?:(?:农历|阴历)\s*)?(.+)$")
                    .unwrap()
            });
            RE.captures(stripped_prefix)
        } {
            (
                parse_chinese_year_digits(caps.get(1)?.as_str())?,
                caps.get(2)?.as_str().to_string(),
            )
        } else if let Some((off, rest)) = limit_year_prefix(stripped_prefix) {
            let rest = rest
                .strip_prefix("农历")
                .or_else(|| rest.strip_prefix("阴历"))
                .unwrap_or(rest);
            (now.year() + off, rest.to_string())
        } else {
            (now.year(), stripped_prefix.to_string())
        }
    };

    // Special case: bare `廿N` or `初N` with no month — inherits lunar
    // month from now. Python treats these as day-only lunar forms.
    if body.starts_with('廿') || body.starts_with('卅') || body.starts_with('念') {
        if let Some((day, tail)) = parse_lunar_day(&body) {
            if tail.trim().is_empty() {
                // Current lunar month.
                use crate::gadget::lunar_solar::solar_to_lunar;
                let (ly, lm, _ld, _is_leap) = solar_to_lunar(now.date())?;
                if let Some(date) = lunar_to_solar(ly, lm, day, false) {
                    return Some(TimeInfo {
                        time_type: "time_point",
                        start: date.and_hms_opt(0, 0, 0)?,
                        end: date.and_hms_opt(23, 59, 59)?,
                        definition: "accurate",
                        ..Default::default()
                    });
                }
            }
        }
    }

    // Gate: require a lunar marker for ambiguous forms. Regular
    // Gregorian-shaped `M月D日/号` + arabic or simple CN numeral day
    // goes to try_absolute_date / try_bare_month_day instead. Python's
    // convention.
    // No gate — Python treats Chinese-numeral-day forms as lunar when
    // they can resolve to a valid lunar date (`二月十五` → lunar Feb 15,
    // `五月廿二` → lunar May 22). Rely on parse_lunar_day's strictness
    // to reject Gregorian-only shapes like `2月15日`.

    // Step 2: parse lunar month + optional day.
    let (lunar_month, is_leap, after_m) = parse_lunar_month(&body)?;
    let after_m = after_m.trim();
    if after_m.is_empty() {
        // Whole lunar month only when the caller clearly meant lunar:
        //   - explicit `农历` / `阴历` marker in the original text, OR
        //   - `闰` leap-month marker, OR
        //   - lunar-only month names (正月 / 腊月 / 冬月 / 梅月).
        let explicit_lunar = text.contains("农历")
            || text.contains("阴历")
            || is_leap
            || body.starts_with("正月")
            || body.starts_with("腊月")
            || body.starts_with("冬月")
            || body.starts_with("梅月");
        if !explicit_lunar {
            return None;
        }
        // For `闰M月` with no explicit year, Python rolls to the closest
        // year that actually contains that leap month (±15 years).
        let effective_year = if is_leap && lunar_to_solar(year, lunar_month, 1, true).is_none() {
            let mut found = None;
            'outer: for delta in 1..=15 {
                for sign in [-1i32, 1] {
                    let y = year + sign * delta;
                    if lunar_to_solar(y, lunar_month, 1, true).is_some() {
                        found = Some(y);
                        break 'outer;
                    }
                }
            }
            match found {
                Some(y) => y,
                None => return None,
            }
        } else {
            year
        };
        // Whole lunar month: span from day 1 to the actual last day.
        // lunar_month_length returns the *regular* month length; the leap
        // month may be 29 or 30 days depending on the year's encoding, so
        // we probe day 30 first and fall back to 29.
        let n = if is_leap {
            if lunar_to_solar(effective_year, lunar_month, 30, true).is_some() {
                30
            } else {
                29
            }
        } else {
            lunar_month_length(effective_year, lunar_month)?
        };
        let start_date = lunar_to_solar(effective_year, lunar_month, 1, is_leap)?;
        let end_date = lunar_to_solar(effective_year, lunar_month, n, is_leap)?;
        return Some(TimeInfo {
            time_type: "time_point",
            start: start_date.and_hms_opt(0, 0, 0)?,
            end: end_date.and_hms_opt(23, 59, 59)?,
            definition: "accurate",
            ..Default::default()
        });
    }
    let (lunar_day, tail) = parse_lunar_day(after_m)?;
    if !tail.trim().is_empty() {
        return None;
    }

    let date = lunar_to_solar(year, lunar_month, lunar_day, is_leap)?;
    let start = date.and_hms_opt(0, 0, 0)?;
    let end = date.and_hms_opt(23, 59, 59)?;
    Some(TimeInfo {
        time_type: "time_point",
        start,
        end,
        definition: "accurate",
        ..Default::default()
    })
}

// ───────────────────────── limit-year + festival (C#30/#32/#34) ───────────

/// Extend `try_fixed_holiday` coverage to `今年/明年/去年/后年 + <festival>`.
fn try_limit_year_festival(text: &str, now: NaiveDateTime) -> Option<TimeInfo> {
    let (off, rest) = limit_year_prefix(text)?;
    let year = now.year() + off;
    // Reuse try_fixed_holiday with a rebuilt "YYYY年<festival>" string.
    let synthesized = format!("{}年{}", year, rest);
    try_fixed_holiday(&synthesized, now)
}

// ───────────────────────── clock-range helper (Round 16) ──────────────────

/// Parse a clock-range string like `上午9点到11点` or `9点到11点`. Returns
/// `(left, right)` as `NaiveTime`.
fn parse_clock_range(s: &str) -> Option<(NaiveTime, NaiveTime)> {
    for sep in ["——", "--", "到", "至", "—", "-", "~", "～"] {
        if let Some(idx) = s.find(sep) {
            let left = s[..idx].trim();
            let right = s[idx + sep.len()..].trim();
            if left.is_empty() || right.is_empty() {
                continue;
            }
            let lc = parse_clock(left)?;
            // Inherit left's AM/PM qualifier when right has none.
            let (rp, _) = split_period(right);
            let (lp, _) = split_period(left);
            let rc = match (rp, lp) {
                (None, Some(lp)) => parse_clock(&format!("{}{}", lp, right))?,
                _ => parse_clock(right)?,
            };
            return Some((lc, rc));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Datelike, NaiveDate, Timelike};

    /// Fixed reference time for deterministic tests.
    fn ref_now() -> NaiveDateTime {
        NaiveDate::from_ymd_opt(2024, 3, 15)
            .unwrap()
            .and_hms_opt(10, 30, 0)
            .unwrap()
    }

    #[test]
    fn absolute_date_cn_full() {
        let t = parse_time("2024年3月5日").unwrap();
        assert_eq!(t.time_type, "time_point");
        assert_eq!(
            t.start,
            NaiveDate::from_ymd_opt(2024, 3, 5)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap()
        );
        assert_eq!(
            t.end,
            NaiveDate::from_ymd_opt(2024, 3, 5)
                .unwrap()
                .and_hms_opt(23, 59, 59)
                .unwrap()
        );
    }

    #[test]
    fn absolute_date_cn_month_only() {
        let t = parse_time("2024年3月").unwrap();
        assert_eq!(t.start.date(), NaiveDate::from_ymd_opt(2024, 3, 1).unwrap());
        assert_eq!(t.end.date(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap());
    }

    #[test]
    fn absolute_date_cn_year_only() {
        let t = parse_time("2024年").unwrap();
        assert_eq!(t.start.date(), NaiveDate::from_ymd_opt(2024, 1, 1).unwrap());
        assert_eq!(t.end.date(), NaiveDate::from_ymd_opt(2024, 12, 31).unwrap());
    }

    #[test]
    fn absolute_date_dash() {
        let t = parse_time("2024-03-05").unwrap();
        assert_eq!(t.start.date(), NaiveDate::from_ymd_opt(2024, 3, 5).unwrap());
    }

    #[test]
    fn absolute_date_slash() {
        let t = parse_time("2024/3/5").unwrap();
        assert_eq!(t.start.date(), NaiveDate::from_ymd_opt(2024, 3, 5).unwrap());
    }

    #[test]
    fn absolute_with_clock() {
        let t = parse_time("2024年3月5日下午3点").unwrap();
        assert_eq!(t.start.hour(), 15);
        assert_eq!(t.start.minute(), 0);
    }

    #[test]
    fn absolute_with_minute() {
        let t = parse_time("2024年3月5日上午8点30分").unwrap();
        assert_eq!(t.start.hour(), 8);
        assert_eq!(t.start.minute(), 30);
    }

    #[test]
    fn relative_today() {
        let t = parse_time_with_ref("今天", ref_now()).unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 3, 15).unwrap()
        );
    }

    #[test]
    fn relative_tomorrow() {
        let t = parse_time_with_ref("明天", ref_now()).unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 3, 16).unwrap()
        );
    }

    #[test]
    fn relative_yesterday() {
        let t = parse_time_with_ref("昨天", ref_now()).unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 3, 14).unwrap()
        );
    }

    #[test]
    fn relative_hou_tian() {
        let t = parse_time_with_ref("后天", ref_now()).unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 3, 17).unwrap()
        );
    }

    #[test]
    fn relative_with_clock() {
        let t = parse_time_with_ref("明天下午3点", ref_now()).unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 3, 16).unwrap()
        );
        assert_eq!(t.start.hour(), 15);
    }

    #[test]
    fn bare_clock() {
        let t = parse_time_with_ref("下午3点", ref_now()).unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 3, 15).unwrap()
        );
        assert_eq!(t.start.hour(), 15);
    }

    #[test]
    fn colon_clock() {
        let t = parse_time_with_ref("08:30", ref_now()).unwrap();
        assert_eq!(t.start.hour(), 8);
        assert_eq!(t.start.minute(), 30);
    }

    #[test]
    fn midnight_qualifier() {
        // 凌晨2点 = 02:00, 下午2点 = 14:00
        let t1 = parse_time_with_ref("凌晨2点", ref_now()).unwrap();
        assert_eq!(t1.start.hour(), 2);
        let t2 = parse_time_with_ref("下午2点", ref_now()).unwrap();
        assert_eq!(t2.start.hour(), 14);
    }

    #[test]
    fn two_digit_year() {
        let t = parse_time("98年3月5日").unwrap();
        assert_eq!(t.start.date().year(), 1998);
    }

    #[test]
    fn invalid_returns_none() {
        assert!(parse_time("not a time").is_none());
        assert!(parse_time("").is_none());
        // Impossible date.
        assert!(parse_time("2024年13月5日").is_none());
        assert!(parse_time("2024-02-30").is_none());
    }

    // ── Stage 2 additions ────────────────────────────────────────────

    #[test]
    fn holiday_guoqing_with_implicit_year() {
        let t = parse_time_with_ref("国庆节", ref_now()).unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 10, 1).unwrap()
        );
    }

    #[test]
    fn holiday_labor_day_explicit_year() {
        let t = parse_time("2023年劳动节").unwrap();
        assert_eq!(t.start.date(), NaiveDate::from_ymd_opt(2023, 5, 1).unwrap());
    }

    #[test]
    fn holiday_alias() {
        let a = parse_time_with_ref("五一", ref_now()).unwrap();
        assert_eq!(a.start.date(), NaiveDate::from_ymd_opt(2024, 5, 1).unwrap());
        let b = parse_time_with_ref("双十一", ref_now()).unwrap();
        assert_eq!(
            b.start.date(),
            NaiveDate::from_ymd_opt(2024, 11, 11).unwrap()
        );
    }

    #[test]
    fn date_range_same_month() {
        let t = parse_time("2024年3月5日到8日").unwrap();
        assert_eq!(t.time_type, "time_span");
        assert_eq!(t.start.date(), NaiveDate::from_ymd_opt(2024, 3, 5).unwrap());
        assert_eq!(t.end.date(), NaiveDate::from_ymd_opt(2024, 3, 8).unwrap());
    }

    #[test]
    fn date_range_cross_month_explicit() {
        // Use the Chinese 到 separator — dash is ambiguous with the date's
        // own dashes ("2024-03-05-2024-04-10") so we require a clear
        // separator.
        let t = parse_time("2024年3月5日到2024年4月10日").unwrap();
        assert_eq!(t.time_type, "time_span");
        assert_eq!(t.start.date(), NaiveDate::from_ymd_opt(2024, 3, 5).unwrap());
        assert_eq!(t.end.date(), NaiveDate::from_ymd_opt(2024, 4, 10).unwrap());
    }

    #[test]
    fn holiday_with_clock() {
        let t = parse_time_with_ref("国庆节下午3点", ref_now()).unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 10, 1).unwrap()
        );
        assert_eq!(t.start.hour(), 15);
    }

    // ── Stage 3 additions ────────────────────────────────────────────

    #[test]
    fn timespan_bare_clock() {
        let t = parse_time_with_ref("8点到12点", ref_now()).unwrap();
        assert_eq!(t.time_type, "time_span");
        assert_eq!(t.start.hour(), 8);
        assert_eq!(t.end.hour(), 12);
    }

    #[test]
    fn timespan_with_period_inheritance() {
        // Right side "5点" should inherit the left period qualifier "下午".
        let t = parse_time_with_ref("下午3点到5点", ref_now()).unwrap();
        assert_eq!(t.start.hour(), 15);
        assert_eq!(t.end.hour(), 17);
    }

    #[test]
    fn timespan_with_relative_day_prefix() {
        let t = parse_time_with_ref("明天下午3点到5点", ref_now()).unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 3, 16).unwrap()
        );
        assert_eq!(t.start.hour(), 15);
        assert_eq!(t.end.hour(), 17);
    }

    #[test]
    fn recurring_weekday() {
        // ref_now is 2024-03-15 = Friday (weekday index 4). 每周一 → next Monday 2024-03-18.
        let t = parse_time_with_ref("每周一", ref_now()).unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 3, 18).unwrap()
        );
    }

    #[test]
    fn recurring_weekday_sunday() {
        // 每周日 from Friday → next Sunday 2024-03-17.
        let t = parse_time_with_ref("每周日", ref_now()).unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 3, 17).unwrap()
        );
    }

    #[test]
    fn recurring_day_of_month() {
        // 每月20号 from 2024-03-15 → 2024-03-20 (still this month).
        let t = parse_time_with_ref("每月20号", ref_now()).unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 3, 20).unwrap()
        );
    }

    #[test]
    fn recurring_day_past_rolls_forward() {
        // 每月10号 from 2024-03-15 → past, roll to 2024-04-10.
        let t = parse_time_with_ref("每月10号", ref_now()).unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 4, 10).unwrap()
        );
    }

    #[test]
    fn recurring_daily_with_clock() {
        // 每天早上8点 → tomorrow at 08:00.
        let t = parse_time_with_ref("每天早上8点", ref_now()).unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 3, 16).unwrap()
        );
        assert_eq!(t.start.hour(), 8);
    }

    // ── Stage 4 additions ────────────────────────────────────────────

    #[test]
    fn delta_three_days_later() {
        let t = parse_time_with_ref("三天后", ref_now()).unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 3, 18).unwrap()
        );
    }

    #[test]
    fn delta_two_weeks_ago() {
        let t = parse_time_with_ref("两周前", ref_now()).unwrap();
        assert_eq!(t.start.date(), NaiveDate::from_ymd_opt(2024, 3, 1).unwrap());
    }

    #[test]
    fn delta_arabic_minutes() {
        // ref_now = 10:30 → +30 min = 11:00.
        let t = parse_time_with_ref("30分钟后", ref_now()).unwrap();
        assert_eq!(t.start.hour(), 11);
        assert_eq!(t.start.minute(), 0);
    }

    #[test]
    fn delta_half_hour_later() {
        let t = parse_time_with_ref("半小时后", ref_now()).unwrap();
        assert_eq!(t.start.hour(), 11);
        assert_eq!(t.start.minute(), 0);
    }

    #[test]
    fn delta_one_year_later() {
        // Round 32: bare `N年后` now emits `time_span` covering the target
        // year (Python parity with pattern #22 `32年前`).
        let t = parse_time_with_ref("一年后", ref_now()).unwrap();
        assert_eq!(t.time_type, "time_span");
        assert_eq!(t.start.date(), NaiveDate::from_ymd_opt(2025, 1, 1).unwrap());
        assert_eq!(t.end.date(), NaiveDate::from_ymd_opt(2025, 12, 31).unwrap());
    }

    #[test]
    fn delta_with_zhihou() {
        // Round 32: `之后` now emits `time_span` (Python parity). The span
        // runs from `now` (2024-03-15) to `now + 3 days`.
        let t = parse_time_with_ref("三天之后", ref_now()).unwrap();
        assert_eq!(t.time_type, "time_span");
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 3, 15).unwrap()
        );
        assert_eq!(t.end.date(), NaiveDate::from_ymd_opt(2024, 3, 18).unwrap());
    }

    #[test]
    fn named_this_week() {
        // ref_now 2024-03-15 = Fri. 本周 → Mon 03-11 .. Sun 03-17.
        // Python classifies named weeks as time_point (referring to a
        // specific named period); we match that convention.
        let t = parse_time_with_ref("本周", ref_now()).unwrap();
        assert_eq!(t.time_type, "time_point");
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 3, 11).unwrap()
        );
        assert_eq!(t.end.date(), NaiveDate::from_ymd_opt(2024, 3, 17).unwrap());
    }

    #[test]
    fn named_last_month() {
        let t = parse_time_with_ref("上个月", ref_now()).unwrap();
        assert_eq!(t.start.date(), NaiveDate::from_ymd_opt(2024, 2, 1).unwrap());
        assert_eq!(t.end.date(), NaiveDate::from_ymd_opt(2024, 2, 29).unwrap());
        // leap year
    }

    #[test]
    fn named_next_quarter() {
        // ref_now month=3 → this quarter = Q1 (Jan..Mar). 下季度 = Q2.
        let t = parse_time_with_ref("下季度", ref_now()).unwrap();
        assert_eq!(t.start.date(), NaiveDate::from_ymd_opt(2024, 4, 1).unwrap());
        assert_eq!(t.end.date(), NaiveDate::from_ymd_opt(2024, 6, 30).unwrap());
    }

    #[test]
    fn named_last_year() {
        let t = parse_time_with_ref("去年", ref_now()).unwrap();
        assert_eq!(t.start.date(), NaiveDate::from_ymd_opt(2023, 1, 1).unwrap());
        assert_eq!(t.end.date(), NaiveDate::from_ymd_opt(2023, 12, 31).unwrap());
    }

    // ── Stage 5 additions ────────────────────────────────────────────

    #[test]
    fn fuzzy_gangcai() {
        let t = parse_time_with_ref("刚才", ref_now()).unwrap();
        assert_eq!(t.definition, "blur");
        // Past ~1 min from 10:30 → around 10:29.
        assert!(t.start < ref_now());
        assert!(t.end < ref_now());
    }

    #[test]
    fn fuzzy_zuijin_recent_window() {
        // "最近" → past ~3 days with 2-day half-width.
        let t = parse_time_with_ref("最近", ref_now()).unwrap();
        assert_eq!(t.definition, "blur");
        // Start is at least 5 days in the past (3 + 2).
        let delta = ref_now() - t.start;
        assert!(delta.num_days() >= 4);
    }

    #[test]
    fn fuzzy_later_today() {
        // "晚些时候" → ~4 hours later with 2-hour half-width.
        let t = parse_time_with_ref("晚些时候", ref_now()).unwrap();
        assert_eq!(t.definition, "blur");
        assert!(t.start > ref_now());
    }

    #[test]
    fn fuzzy_longest_match_wins() {
        // "不久之前" must win over "不久" or "不久前".
        let t = parse_time_with_ref("不久之前", ref_now()).unwrap();
        assert_eq!(t.definition, "blur");
    }

    #[test]
    fn fuzzy_nonmatch_returns_none() {
        assert!(parse_time_with_ref("abcdef", ref_now()).is_none());
    }

    #[test]
    fn fuzzy_mashangjiu_is_near_future() {
        let t = parse_time_with_ref("马上", ref_now()).unwrap();
        assert_eq!(t.definition, "blur");
        // Mid-point is ~60 seconds from now; always > now in start..end range.
        assert!(t.end > ref_now());
    }

    // ── Stage 6 additions (lunar holidays) ───────────────────────────

    #[test]
    fn lunar_spring_festival_2024() {
        let t = parse_time_with_ref("春节", ref_now()).unwrap();
        // Year inferred from ref_now (2024) → 2024-02-10.
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 2, 10).unwrap()
        );
    }

    #[test]
    fn lunar_spring_festival_explicit_year() {
        let t = parse_time("2025年春节").unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2025, 1, 29).unwrap()
        );
    }

    #[test]
    fn lunar_mid_autumn_2024() {
        let t = parse_time_with_ref("中秋节", ref_now()).unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 9, 17).unwrap()
        );
    }

    #[test]
    fn lunar_mid_autumn_alias_without_jie() {
        let t = parse_time_with_ref("中秋", ref_now()).unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 9, 17).unwrap()
        );
    }

    #[test]
    fn lunar_dragon_boat_2023() {
        let t = parse_time("2023年端午节").unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2023, 6, 22).unwrap()
        );
    }

    #[test]
    fn lunar_chongyang_2024() {
        let t = parse_time_with_ref("重阳节", ref_now()).unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 10, 11).unwrap()
        );
    }

    #[test]
    fn lunar_new_year_eve_2024() {
        let t = parse_time_with_ref("除夕", ref_now()).unwrap();
        assert_eq!(t.start.date(), NaiveDate::from_ymd_opt(2024, 2, 9).unwrap());
    }

    #[test]
    fn lunar_chunjie_with_clock() {
        let t = parse_time_with_ref("春节下午3点", ref_now()).unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2024, 2, 10).unwrap()
        );
        assert_eq!(t.start.hour(), 15);
    }

    #[test]
    fn lunar_out_of_range_returns_none() {
        // Year 1899 is outside our 1900-2100 table.
        assert!(parse_time("1899年春节").is_none());
        assert!(parse_time("2101年春节").is_none());
    }

    #[test]
    fn lunar_extended_range_2019() {
        // Previously out of range (old 2020-2035 table), now covered by
        // the full 1900-2100 lunar converter.
        let t = parse_time("2019年春节").unwrap();
        assert_eq!(t.start.date(), NaiveDate::from_ymd_opt(2019, 2, 5).unwrap());
    }

    // ── Round 15 — Chinese-numeral year + clock suffixes ──────────────

    #[test]
    fn chinese_year_lingsan() {
        // 零三年 → 2003.
        let t = parse_time("零三年3月5日").unwrap();
        use chrono::Datelike;
        assert_eq!(t.start.year(), 2003);
    }

    #[test]
    fn chinese_year_full_form() {
        // 二零二四年 → 2024. 二〇二四年 likewise.
        let t1 = parse_time("二零二四年3月5日").unwrap();
        let t2 = parse_time("二〇二四年3月5日").unwrap();
        use chrono::Datelike;
        assert_eq!(t1.start.year(), 2024);
        assert_eq!(t2.start.year(), 2024);
    }

    #[test]
    fn clock_bamian() {
        // 8点半 = 08:30
        let now = ref_now();
        let t = parse_time_with_ref("8点半", now).unwrap();
        assert_eq!(t.start.hour(), 8);
        assert_eq!(t.start.minute(), 30);
    }

    #[test]
    fn clock_wanshang_bamian() {
        // 晚上8点半 = 20:30
        let t = parse_time_with_ref("晚上8点半", ref_now()).unwrap();
        assert_eq!(t.start.hour(), 20);
        assert_eq!(t.start.minute(), 30);
    }

    #[test]
    fn clock_yike_sanke() {
        let t = parse_time_with_ref("下午3点一刻", ref_now()).unwrap();
        assert_eq!(t.start.hour(), 15);
        assert_eq!(t.start.minute(), 15);

        let t = parse_time_with_ref("下午3点三刻", ref_now()).unwrap();
        assert_eq!(t.start.hour(), 15);
        assert_eq!(t.start.minute(), 45);
    }

    // ── The user-reported regression ──────────────────────────────────

    #[test]
    fn user_example_lingsan_yuanxiao_wanshang_bamian() {
        // "零三年元宵节晚上8点半" → 2003-02-15 20:30:00
        let t = parse_time("零三年元宵节晚上8点半").unwrap();
        assert_eq!(
            t.start.date(),
            NaiveDate::from_ymd_opt(2003, 2, 15).unwrap()
        );
        assert_eq!(t.start.hour(), 20);
        assert_eq!(t.start.minute(), 30);
    }
}
