//! Rustler NIF bindings exposing jionlp-core to Elixir.
//!
//! Keep this file a thin adapter: validate inputs, convert types, dispatch.
//! Any real logic belongs in `jionlp-core`.

use jionlp_core as core;
use rustler::{Atom, NifResult, NifStruct};

mod atoms {
    rustler::atoms! {
        ok,
        error,
        coarse,
        fine,
        char_mode = "char",
        word_mode = "word",
        simplified,
        traditional,
    }
}

// ─────────────────────── Shared structs for Elixir ──────────────────────────

/// Matches `%JioNLP.Extracted{text: "...", offset: {start, end}}` on the
/// Elixir side.
#[derive(NifStruct)]
#[module = "JioNLP.Extracted"]
struct NifExtracted {
    text: String,
    offset: (u64, u64),
}

impl From<core::Extracted> for NifExtracted {
    fn from(e: core::Extracted) -> Self {
        NifExtracted {
            text: e.text,
            offset: (e.offset.0 as u64, e.offset.1 as u64),
        }
    }
}

#[derive(NifStruct)]
#[module = "JioNLP.PlateInfo"]
struct NifPlateInfo {
    car_loc: String,
    car_type: String,
    car_size: Option<String>,
}

impl From<core::PlateInfo> for NifPlateInfo {
    fn from(p: core::PlateInfo) -> Self {
        NifPlateInfo {
            car_loc: p.car_loc,
            car_type: p.car_type,
            car_size: p.car_size.map(String::from),
        }
    }
}

#[derive(NifStruct)]
#[module = "JioNLP.IdCardInfo"]
struct NifIdCardInfo {
    province: String,
    city: Option<String>,
    county: Option<String>,
    birth_year: String,
    birth_month: String,
    birth_day: String,
    gender: String,
    check_code: String,
}

impl From<core::IdCardInfo> for NifIdCardInfo {
    fn from(i: core::IdCardInfo) -> Self {
        NifIdCardInfo {
            province: i.province,
            city: i.city,
            county: i.county,
            birth_year: i.birth_year,
            birth_month: i.birth_month,
            birth_day: i.birth_day,
            gender: i.gender.to_string(),
            check_code: i.check_code,
        }
    }
}

#[derive(NifStruct)]
#[module = "JioNLP.RadicalInfo"]
struct NifRadicalInfo {
    char: String,
    radical: String,
    structure: String,
    corner_coding: String,
    stroke_order: String,
    wubi_coding: String,
}

#[derive(NifStruct)]
#[module = "JioNLP.PhoneInfo"]
struct NifPhoneInfo {
    number: String,
    province: Option<String>,
    city: Option<String>,
    phone_type: String,
    operator: Option<String>,
}

impl From<core::PhoneInfo> for NifPhoneInfo {
    fn from(p: core::PhoneInfo) -> Self {
        NifPhoneInfo {
            number: p.number,
            province: p.province,
            city: p.city,
            phone_type: p.phone_type.to_string(),
            operator: p.operator,
        }
    }
}

#[derive(NifStruct)]
#[module = "JioNLP.LocationMatch"]
struct NifLocationMatch {
    name: String,
    level: String,
    offset: (u64, u64),
}

impl From<core::LocationMatch> for NifLocationMatch {
    fn from(m: core::LocationMatch) -> Self {
        NifLocationMatch {
            name: m.name,
            level: match m.level {
                core::LocationLevel::Province => "province".to_string(),
                core::LocationLevel::City => "city".to_string(),
                core::LocationLevel::County => "county".to_string(),
            },
            offset: (m.offset.0 as u64, m.offset.1 as u64),
        }
    }
}

#[derive(NifStruct)]
#[module = "JioNLP.ParsedLocation"]
struct NifParsedLocation {
    province: Option<String>,
    city: Option<String>,
    county: Option<String>,
}

impl From<core::ParsedLocation> for NifParsedLocation {
    fn from(p: core::ParsedLocation) -> Self {
        NifParsedLocation {
            province: p.province,
            city: p.city,
            county: p.county,
        }
    }
}

/// Full Python-parity location parse result — returned by
/// `JioNLP.parse_location_full/3`. `town` / `village` are `Some` only when
/// caller passes `town_village=true` and there's a matching 4/5-level
/// dictionary entry under the admin triple.
#[derive(NifStruct)]
#[module = "JioNLP.LocationParseResult"]
struct NifLocationParseResult {
    province: Option<String>,
    city: Option<String>,
    county: Option<String>,
    detail: String,
    full_location: String,
    orig_location: String,
    town: Option<String>,
    village: Option<String>,
}

impl From<core::LocationParseResult> for NifLocationParseResult {
    fn from(r: core::LocationParseResult) -> Self {
        NifLocationParseResult {
            province: r.province,
            city: r.city,
            county: r.county,
            detail: r.detail,
            full_location: r.full_location,
            orig_location: r.orig_location,
            town: r.town,
            village: r.village,
        }
    }
}

#[derive(NifStruct)]
#[module = "JioNLP.MoneyInfo"]
struct NifMoneyInfo {
    num: f64,
    case: String,
    definition: String,
    end_num: Option<f64>,
}

impl From<core::MoneyInfo> for NifMoneyInfo {
    fn from(m: core::MoneyInfo) -> Self {
        NifMoneyInfo {
            num: m.num,
            case: m.case,
            definition: m.definition.to_string(),
            end_num: m.end_num,
        }
    }
}

#[derive(NifStruct)]
#[module = "JioNLP.PinyinDetail"]
struct NifPinyinDetail {
    consonant: String,
    vowel: String,
    tone: String,
}

impl From<core::PinyinDetail> for NifPinyinDetail {
    fn from(d: core::PinyinDetail) -> Self {
        NifPinyinDetail {
            consonant: d.consonant,
            vowel: d.vowel,
            tone: d.tone,
        }
    }
}

/// Each delta unit is either `nil`, `{:single, v}` (exact), or `{:range,
/// lo, hi}` (fuzzy). Rustler serializes Option<enum> as `nil | tuple`.
#[derive(rustler::NifTaggedEnum)]
enum NifDeltaValue {
    Single(f64),
    Range(f64, f64),
}

impl From<core::DeltaValue> for NifDeltaValue {
    fn from(v: core::DeltaValue) -> Self {
        match v {
            core::DeltaValue::Single(n) => NifDeltaValue::Single(n),
            core::DeltaValue::Range(lo, hi) => NifDeltaValue::Range(lo, hi),
        }
    }
}

#[derive(NifStruct)]
#[module = "JioNLP.TimeDelta"]
struct NifTimeDelta {
    year: Option<NifDeltaValue>,
    month: Option<NifDeltaValue>,
    day: Option<NifDeltaValue>,
    hour: Option<NifDeltaValue>,
    minute: Option<NifDeltaValue>,
    second: Option<NifDeltaValue>,
    workday: Option<NifDeltaValue>,
    zero: bool,
}

impl From<core::TimeDelta> for NifTimeDelta {
    fn from(d: core::TimeDelta) -> Self {
        NifTimeDelta {
            year: d.year.map(Into::into),
            month: d.month.map(Into::into),
            day: d.day.map(Into::into),
            hour: d.hour.map(Into::into),
            minute: d.minute.map(Into::into),
            second: d.second.map(Into::into),
            workday: d.workday.map(Into::into),
            zero: d.zero,
        }
    }
}

#[derive(NifStruct)]
#[module = "JioNLP.TimePeriod"]
struct NifTimePeriod {
    delta: NifTimeDelta,
    /// List of `(start_iso, end_iso)` tuples — each anchor within one cycle.
    point_time: Vec<(String, String)>,
    point_string: String,
}

impl From<core::TimePeriodInfo> for NifTimePeriod {
    fn from(p: core::TimePeriodInfo) -> Self {
        NifTimePeriod {
            delta: p.delta.into(),
            point_time: p
                .point_time
                .into_iter()
                .map(|(s, e)| {
                    (
                        s.format("%Y-%m-%dT%H:%M:%S").to_string(),
                        e.format("%Y-%m-%dT%H:%M:%S").to_string(),
                    )
                })
                .collect(),
            point_string: p.point_string,
        }
    }
}

#[derive(NifStruct)]
#[module = "JioNLP.TimeInfo"]
struct NifTimeInfo {
    time_type: String,
    // ISO 8601 "YYYY-MM-DDTHH:MM:SS" strings for clean Elixir-side parsing.
    // `start`/`end` are zero-valued sentinels for `time_delta` — callers
    // should read `delta` / `period` for those variants.
    start: String,
    end: String,
    definition: String,
    delta: Option<NifTimeDelta>,
    period: Option<NifTimePeriod>,
}

impl From<core::TimeInfo> for NifTimeInfo {
    fn from(t: core::TimeInfo) -> Self {
        NifTimeInfo {
            time_type: t.time_type.to_string(),
            start: t.start.format("%Y-%m-%dT%H:%M:%S").to_string(),
            end: t.end.format("%Y-%m-%dT%H:%M:%S").to_string(),
            definition: t.definition.to_string(),
            delta: t.delta.map(Into::into),
            period: t.period.map(Into::into),
        }
    }
}

#[derive(NifStruct)]
#[module = "JioNLP.KeyPhrase"]
struct NifKeyPhrase {
    phrase: String,
    weight: f64,
}

impl From<core::KeyPhrase> for NifKeyPhrase {
    fn from(k: core::KeyPhrase) -> Self {
        NifKeyPhrase {
            phrase: k.phrase,
            weight: k.weight,
        }
    }
}

#[derive(NifStruct)]
#[module = "JioNLP.SummarySentence"]
struct NifSummarySentence {
    text: String,
    score: f64,
    position: u64,
}

impl From<core::SummarySentence> for NifSummarySentence {
    fn from(s: core::SummarySentence) -> Self {
        NifSummarySentence {
            text: s.text,
            score: s.score,
            position: s.position as u64,
        }
    }
}

#[derive(NifStruct)]
#[module = "JioNLP.NerEntity"]
struct NifNerEntity {
    text: String,
    entity_type: String,
    offset: (u64, u64),
}

impl From<core::NerEntity> for NifNerEntity {
    fn from(e: core::NerEntity) -> Self {
        NifNerEntity {
            text: e.text,
            entity_type: e.entity_type,
            offset: (e.offset.0 as u64, e.offset.1 as u64),
        }
    }
}

#[derive(NifStruct)]
#[module = "JioNLP.NamedEntity"]
struct NifNamedEntity {
    text: String,
    entity_type: String,
    offset: (u64, u64),
}

impl From<core::NamedEntity> for NifNamedEntity {
    fn from(e: core::NamedEntity) -> Self {
        NifNamedEntity {
            text: e.text,
            entity_type: e.entity_type,
            offset: (e.offset.0 as u64, e.offset.1 as u64),
        }
    }
}

impl From<NifNamedEntity> for core::NamedEntity {
    fn from(e: NifNamedEntity) -> Self {
        core::NamedEntity {
            text: e.text,
            entity_type: e.entity_type,
            offset: (e.offset.0 as usize, e.offset.1 as usize),
        }
    }
}

#[derive(NifStruct)]
#[module = "JioNLP.EntityAugmented"]
struct NifEntityAugmented {
    text: String,
    entities: Vec<NifNamedEntity>,
}

impl From<core::EntityAugmented> for NifEntityAugmented {
    fn from(a: core::EntityAugmented) -> Self {
        NifEntityAugmented {
            text: a.text,
            entities: a.entities.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<core::RadicalInfo> for NifRadicalInfo {
    fn from(r: core::RadicalInfo) -> Self {
        NifRadicalInfo {
            char: r.char.to_string(),
            radical: r.radical,
            structure: r.structure,
            corner_coding: r.corner_coding,
            stroke_order: r.stroke_order,
            wubi_coding: r.wubi_coding,
        }
    }
}

// ─────────────────────────────── NIFs ────────────────────────────────────────

#[rustler::nif]
fn init_dictionaries(dict_path: String) -> NifResult<Atom> {
    core::dict::init_from_path(&dict_path)
        .map(|_| atoms::ok())
        .map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))
}

// ── gadget ──────────────────────────────────────────────────────────────────

#[rustler::nif]
fn split_sentence(text: String, criterion: Atom) -> NifResult<Vec<String>> {
    let c = if criterion == atoms::fine() {
        core::Criterion::Fine
    } else {
        core::Criterion::Coarse
    };
    Ok(core::split_sentence(&text, c))
}

#[rustler::nif]
fn remove_stopwords(words: Vec<String>, save_negative: bool) -> NifResult<Vec<String>> {
    core::remove_stopwords(
        &words,
        core::RemoveOpts {
            save_negative_words: save_negative,
        },
    )
    .map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))
}

#[rustler::nif]
fn tra2sim(text: String, mode: Atom) -> NifResult<String> {
    let m = if mode == atoms::word_mode() {
        core::TsMode::Word
    } else {
        core::TsMode::Char
    };
    core::tra2sim(&text, m).map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))
}

#[rustler::nif]
fn sim2tra(text: String, mode: Atom) -> NifResult<String> {
    let m = if mode == atoms::word_mode() {
        core::TsMode::Word
    } else {
        core::TsMode::Char
    };
    core::sim2tra(&text, m).map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))
}

#[rustler::nif]
fn parse_motor_vehicle_licence_plate(plate: String) -> NifResult<Option<NifPlateInfo>> {
    Ok(core::parse_motor_vehicle_licence_plate(&plate).map(Into::into))
}

#[rustler::nif]
fn parse_id_card(id_card: String) -> NifResult<Option<NifIdCardInfo>> {
    core::parse_id_card(&id_card)
        .map(|opt| opt.map(Into::into))
        .map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))
}

#[rustler::nif]
fn char_radical(text: String) -> NifResult<Vec<NifRadicalInfo>> {
    core::char_radical(&text)
        .map(|v| v.into_iter().map(Into::into).collect())
        .map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))
}

// ── num ↔ char ─────────────────────────────────────────────────────────────

#[rustler::nif]
fn num2char(num: String, style: Atom) -> NifResult<String> {
    let s = if style == atoms::traditional() {
        core::NumStyle::Traditional
    } else {
        core::NumStyle::Simplified
    };
    core::num2char(&num, s).map_err(|e| rustler::Error::Term(Box::new(e)))
}

#[rustler::nif]
fn char2num(text: String) -> NifResult<f64> {
    core::char2num(&text).map_err(|e| rustler::Error::Term(Box::new(e)))
}

// ── html_cleansing ─────────────────────────────────────────────────────────

#[rustler::nif]
fn remove_html_tag(text: String) -> NifResult<String> {
    Ok(core::remove_html_tag(&text))
}

#[rustler::nif]
fn clean_html(text: String) -> NifResult<String> {
    Ok(core::clean_html(&text))
}

#[rustler::nif]
fn remove_redundant_char(text: String, custom: Option<String>) -> NifResult<String> {
    Ok(core::remove_redundant_char(&text, custom.as_deref()))
}

// ── phone_location / location_recognizer ──────────────────────────────────

#[rustler::nif]
fn phone_location(text: String) -> NifResult<NifPhoneInfo> {
    core::phone_location(&text)
        .map(Into::into)
        .map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))
}

#[rustler::nif]
fn cell_phone_location(text: String, digits: String) -> NifResult<NifPhoneInfo> {
    core::cell_phone_location(&text, &digits)
        .map(Into::into)
        .map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))
}

#[rustler::nif]
fn landline_phone_location(text: String) -> NifResult<NifPhoneInfo> {
    core::landline_phone_location(&text)
        .map(Into::into)
        .map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))
}

#[rustler::nif]
fn recognize_location(text: String) -> NifResult<Vec<NifLocationMatch>> {
    core::recognize_location(&text)
        .map(|v| v.into_iter().map(Into::into).collect())
        .map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))
}

#[rustler::nif]
fn parse_location(text: String) -> NifResult<NifParsedLocation> {
    core::parse_location(&text)
        .map(Into::into)
        .map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))
}

#[rustler::nif]
fn parse_location_full(
    text: String,
    town_village: bool,
    change2new: bool,
) -> NifResult<NifLocationParseResult> {
    core::parse_location_full(&text, town_village, change2new)
        .map(Into::into)
        .map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))
}

// ── pinyin ─────────────────────────────────────────────────────────────────

#[rustler::nif]
fn pinyin_standard(text: String) -> NifResult<Vec<String>> {
    let out = core::pinyin(&text, core::PinyinFormat::Standard)
        .map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))?;
    Ok(out
        .into_iter()
        .map(|e| match e {
            core::PinyinEntry::Standard(s) => s,
            _ => String::new(),
        })
        .collect())
}

#[rustler::nif]
fn pinyin_simple(text: String) -> NifResult<Vec<String>> {
    let out = core::pinyin(&text, core::PinyinFormat::Simple)
        .map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))?;
    Ok(out
        .into_iter()
        .map(|e| match e {
            core::PinyinEntry::Simple(s) => s,
            _ => String::new(),
        })
        .collect())
}

#[rustler::nif]
fn pinyin_detail(text: String) -> NifResult<Vec<NifPinyinDetail>> {
    let out = core::pinyin(&text, core::PinyinFormat::Detail)
        .map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))?;
    Ok(out
        .into_iter()
        .map(|e| match e {
            core::PinyinEntry::Detail(d) => d.into(),
            _ => NifPinyinDetail {
                consonant: String::new(),
                vowel: String::new(),
                tone: String::new(),
            },
        })
        .collect())
}

// ── parse_money ────────────────────────────────────────────────────────────

#[rustler::nif]
fn parse_money(text: String) -> NifResult<Option<NifMoneyInfo>> {
    Ok(core::parse_money(&text).map(Into::into))
}

#[rustler::nif]
fn parse_money_with_default(text: String, default_unit: String) -> NifResult<Option<NifMoneyInfo>> {
    Ok(core::parse_money_with_default(&text, &default_unit).map(Into::into))
}

// ── parse_time ─────────────────────────────────────────────────────────────

#[rustler::nif]
fn parse_time(text: String) -> NifResult<Option<NifTimeInfo>> {
    Ok(core::parse_time(&text).map(Into::into))
}

/// Parse relative to an explicit reference time given as an ISO string
/// "YYYY-MM-DDTHH:MM:SS" or "YYYY-MM-DD HH:MM:SS". Invalid ref returns error.
#[rustler::nif]
fn parse_time_with_ref(text: String, reference_iso: String) -> NifResult<Option<NifTimeInfo>> {
    let fmts = ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"];
    for fmt in &fmts {
        if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(&reference_iso, fmt) {
            return Ok(core::parse_time_with_ref(&text, dt).map(Into::into));
        }
    }
    Err(rustler::Error::Term(Box::new(
        "reference_iso must be 'YYYY-MM-DDTHH:MM:SS' or 'YYYY-MM-DD HH:MM:SS'".to_string(),
    )))
}

// ── simhash ────────────────────────────────────────────────────────────────

#[rustler::nif]
fn simhash(text: String) -> u64 {
    core::simhash(&text)
}

#[rustler::nif]
fn simhash_ngram(text: String, n: u32) -> u64 {
    core::simhash_ngram(&text, n as usize)
}

#[rustler::nif]
fn hamming_distance(a: u64, b: u64) -> u32 {
    core::hamming_distance(a, b)
}

#[rustler::nif]
fn simhash_similarity(a: u64, b: u64) -> f64 {
    core::simhash_similarity(a, b)
}

// ── keyphrase ──────────────────────────────────────────────────────────────

#[rustler::nif]
fn extract_keyphrase(
    text: String,
    top_k: u32,
    min_n: u32,
    max_n: u32,
) -> NifResult<Vec<NifKeyPhrase>> {
    core::extract_keyphrase(&text, top_k as usize, min_n as usize, max_n as usize)
        .map(|v| v.into_iter().map(Into::into).collect())
        .map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))
}

#[rustler::nif]
fn extract_keyphrase_textrank(
    text: String,
    top_k: u32,
    min_n: u32,
    max_n: u32,
) -> NifResult<Vec<NifKeyPhrase>> {
    core::extract_keyphrase_textrank(&text, top_k as usize, min_n as usize, max_n as usize)
        .map(|v| v.into_iter().map(Into::into).collect())
        .map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))
}

// ── sentiment / summary / textaug ─────────────────────────────────────────

#[rustler::nif]
fn sentiment_score(text: String) -> NifResult<f64> {
    core::sentiment_score(&text).map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))
}

#[rustler::nif]
fn extract_summary(text: String, top_k: u32) -> NifResult<Vec<NifSummarySentence>> {
    core::extract_summary(&text, top_k as usize)
        .map(|v| v.into_iter().map(Into::into).collect())
        .map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))
}

#[rustler::nif]
fn extract_summary_by_length(text: String, max_chars: u32) -> NifResult<Vec<NifSummarySentence>> {
    core::extract_summary_by_length(&text, max_chars as usize)
        .map(|v| v.into_iter().map(Into::into).collect())
        .map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))
}

#[rustler::nif]
fn swap_char_position(
    text: String,
    n: u32,
    swap_ratio: f64,
    seed: u64,
    scale: f64,
) -> NifResult<Vec<String>> {
    Ok(core::swap_char_position(
        &text, n as usize, swap_ratio, seed, scale,
    ))
}

#[rustler::nif]
fn random_add_delete(
    text: String,
    n: u32,
    add_ratio: f64,
    delete_ratio: f64,
    seed: u64,
) -> NifResult<Vec<String>> {
    Ok(core::random_add_delete(
        &text,
        n as usize,
        add_ratio,
        delete_ratio,
        seed,
    ))
}

#[rustler::nif]
fn homophone_substitution(
    text: String,
    n: u32,
    sub_ratio: f64,
    seed: u64,
) -> NifResult<Vec<String>> {
    core::homophone_substitution(&text, n as usize, sub_ratio, seed)
        .map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))
}

#[rustler::nif]
fn replace_entity(
    text: String,
    entities: Vec<NifNamedEntity>,
    // replacements come as [(entity_type, [(name, weight), ...]), ...]
    replacements: Vec<(String, Vec<(String, f64)>)>,
    n: u32,
    replace_ratio: f64,
    seed: u64,
) -> NifResult<Vec<NifEntityAugmented>> {
    let entities: Vec<core::NamedEntity> = entities.into_iter().map(Into::into).collect();
    let replacements: rustc_hash::FxHashMap<String, Vec<(String, f64)>> =
        replacements.into_iter().collect();
    let out = core::replace_entity(
        &text,
        &entities,
        &replacements,
        n as usize,
        replace_ratio,
        seed,
    );
    Ok(out.into_iter().map(Into::into).collect())
}

// ── algorithm/ner (stateless convenience) ─────────────────────────────────

#[rustler::nif]
fn recognize_entities(
    text: String,
    lexicon: Vec<(String, Vec<String>)>,
) -> NifResult<Vec<NifNerEntity>> {
    let ner = core::LexiconNer::new(lexicon).map_err(|e| rustler::Error::Term(Box::new(e)))?;
    Ok(ner.recognize(&text).into_iter().map(Into::into).collect())
}

// ── algorithm/bpe + summary MMR ───────────────────────────────────────────

#[rustler::nif]
fn bpe_encode(text: String) -> String {
    core::bpe_encode(&text)
}

#[rustler::nif]
fn bpe_decode(encoded: String) -> String {
    core::bpe_decode(&encoded)
}

#[rustler::nif]
fn extract_summary_mmr(
    text: String,
    top_k: u32,
    lambda: f64,
) -> NifResult<Vec<NifSummarySentence>> {
    core::extract_summary_mmr(&text, top_k as usize, lambda)
        .map(|v| v.into_iter().map(Into::into).collect())
        .map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))
}

// ── rule::extractor ─────────────────────────────────────────────────────────

fn to_nif_vec(v: Vec<core::Extracted>) -> Vec<NifExtracted> {
    v.into_iter().map(Into::into).collect()
}

#[rustler::nif]
fn extract_email(text: String) -> NifResult<Vec<NifExtracted>> {
    Ok(to_nif_vec(core::extract_email(&text)))
}
#[rustler::nif]
fn extract_cell_phone(text: String) -> NifResult<Vec<NifExtracted>> {
    Ok(to_nif_vec(core::extract_cell_phone(&text)))
}
#[rustler::nif]
fn extract_landline_phone(text: String) -> NifResult<Vec<NifExtracted>> {
    Ok(to_nif_vec(core::extract_landline_phone(&text)))
}
#[rustler::nif]
fn extract_phone_number(text: String) -> NifResult<Vec<NifExtracted>> {
    Ok(to_nif_vec(core::extract_phone_number(&text)))
}
#[rustler::nif]
fn extract_ip_address(text: String) -> NifResult<Vec<NifExtracted>> {
    Ok(to_nif_vec(core::extract_ip_address(&text)))
}
#[rustler::nif]
fn extract_id_card(text: String) -> NifResult<Vec<NifExtracted>> {
    Ok(to_nif_vec(core::extract_id_card(&text)))
}
#[rustler::nif]
fn extract_url(text: String) -> NifResult<Vec<NifExtracted>> {
    Ok(to_nif_vec(core::extract_url(&text)))
}
#[rustler::nif]
fn extract_qq(text: String) -> NifResult<Vec<NifExtracted>> {
    Ok(to_nif_vec(core::extract_qq(&text)))
}
#[rustler::nif]
fn extract_motor_vehicle_licence_plate(text: String) -> NifResult<Vec<NifExtracted>> {
    Ok(to_nif_vec(core::extract_motor_vehicle_licence_plate(&text)))
}
#[rustler::nif]
fn extract_chinese(text: String) -> NifResult<Vec<String>> {
    Ok(core::extract_chinese(&text))
}
#[rustler::nif]
fn extract_parentheses(text: String, table: String) -> NifResult<Vec<NifExtracted>> {
    Ok(to_nif_vec(core::extract_parentheses(&text, &table)))
}

// ── rule::checker ───────────────────────────────────────────────────────────

#[rustler::nif]
fn check_any_chinese_char(text: String) -> bool {
    core::check_any_chinese_char(&text)
}
#[rustler::nif]
fn check_all_chinese_char(text: String) -> bool {
    core::check_all_chinese_char(&text)
}
#[rustler::nif]
fn check_any_arabic_num(text: String) -> bool {
    core::check_any_arabic_num(&text)
}
#[rustler::nif]
fn check_all_arabic_num(text: String) -> bool {
    core::check_all_arabic_num(&text)
}
#[rustler::nif]
fn check_id_card(text: String) -> bool {
    core::check_id_card(&text)
}
#[rustler::nif]
fn check_cell_phone(text: String) -> bool {
    core::check_cell_phone(&text)
}
#[rustler::nif]
fn check_motor_vehicle_licence_plate(text: String) -> bool {
    core::check_motor_vehicle_licence_plate(&text)
}

// ── Round 35 — newly-exposed APIs ─────────────────────────────────────────

#[rustler::nif]
fn get_china_province_alias(name: String) -> Option<String> {
    core::get_china_province_alias(&name).map(String::from)
}

#[rustler::nif]
fn get_china_city_alias(name: String, dismiss_diqu: bool, dismiss_meng: bool) -> Option<String> {
    core::get_china_city_alias(&name, dismiss_diqu, dismiss_meng)
}

#[rustler::nif]
fn get_china_county_alias(name: String, dismiss_qi: bool) -> Option<String> {
    core::get_china_county_alias(&name, dismiss_qi)
}

#[rustler::nif]
fn get_china_town_alias(name: String) -> Option<String> {
    core::get_china_town_alias(&name)
}

#[rustler::nif]
fn is_person_name(text: String) -> bool {
    core::is_person_name(&text)
}

#[rustler::nif]
fn solar_to_lunar(iso_date: String) -> NifResult<(i32, u32, u32, bool)> {
    let d = chrono::NaiveDate::parse_from_str(&iso_date, "%Y-%m-%d")
        .map_err(|e| rustler::Error::Term(Box::new(format!("parse date: {e}"))))?;
    core::solar_to_lunar(d)
        .ok_or_else(|| rustler::Error::Term(Box::new("date out of 1900-2100 range".to_string())))
}

#[rustler::nif]
fn lunar_to_solar(
    lunar_year: i32,
    lunar_month: u32,
    lunar_day: u32,
    leap_month: bool,
) -> Option<String> {
    core::lunar_to_solar(lunar_year, lunar_month, lunar_day, leap_month)
        .map(|d| d.format("%Y-%m-%d").to_string())
}

#[rustler::nif]
fn remove_email(text: String) -> String {
    core::remove_email(&text)
}
#[rustler::nif]
fn remove_url(text: String) -> String {
    core::remove_url(&text)
}
#[rustler::nif]
fn remove_phone_number(text: String) -> String {
    core::remove_phone_number(&text)
}
#[rustler::nif]
fn remove_ip_address(text: String) -> String {
    core::remove_ip_address(&text)
}
#[rustler::nif]
fn remove_id_card(text: String) -> String {
    core::remove_id_card(&text)
}
#[rustler::nif]
fn remove_qq(text: String) -> String {
    core::remove_qq(&text)
}
#[rustler::nif]
fn remove_parentheses(text: String) -> String {
    core::remove_parentheses(&text, None)
}
#[rustler::nif]
fn remove_exception_char(text: String) -> String {
    core::remove_exception_char(&text)
}
#[rustler::nif]
fn replace_email(text: String, placeholder: String) -> String {
    core::replace_email(&text, &placeholder)
}
#[rustler::nif]
fn replace_url(text: String, placeholder: String) -> String {
    core::replace_url(&text, &placeholder)
}
#[rustler::nif]
fn replace_phone_number(text: String, placeholder: String) -> String {
    core::replace_phone_number(&text, &placeholder)
}
#[rustler::nif]
fn replace_chinese(text: String, placeholder: String) -> String {
    core::replace_chinese(&text, &placeholder)
}
#[rustler::nif]
fn convert_full2half(text: String) -> String {
    core::convert_full2half(&text)
}
#[rustler::nif]
fn extract_wechat_id(text: String) -> Vec<String> {
    core::extract_wechat_id(&text)
}

#[rustler::nif]
fn idiom_next_by_char(seed: u64, cur: String, with_prob: bool) -> NifResult<Option<String>> {
    let game = core::IdiomSolitaireGame::new(seed);
    game.next_by_char(&cur, with_prob)
        .map_err(|e| rustler::Error::Term(Box::new(format!("{e}"))))
}

rustler::init!("Elixir.JioNLP.Native");
