//! Dictionary loading infrastructure.
//!
//! Dictionaries live as plain-text files on disk. The application must call
//! [`init_from_path`] once on startup with the directory that contains them
//! (typically the Python project's `jionlp/dictionary/` folder, or a copy).
//!
//! After initialization, individual loaders ([`stopwords`], [`tra2sim_char`],
//! etc.) return cached references to their dictionary — zero IO on subsequent
//! calls, zero locks on concurrent reads.

use crate::{Error, Result};
use once_cell::sync::OnceCell;
use rustc_hash::{FxHashMap, FxHashSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Root path containing all dictionary files. Set by [`init_from_path`].
static DICT_ROOT: OnceCell<PathBuf> = OnceCell::new();

static STOPWORDS: OnceCell<FxHashSet<String>> = OnceCell::new();
static NEGATIVE_WORDS: OnceCell<FxHashSet<String>> = OnceCell::new();
static TRA2SIM_CHAR: OnceCell<FxHashMap<String, String>> = OnceCell::new();
static SIM2TRA_CHAR: OnceCell<FxHashMap<String, String>> = OnceCell::new();
static TRA2SIM_WORD: OnceCell<FxHashMap<String, String>> = OnceCell::new();
static SIM2TRA_WORD: OnceCell<FxHashMap<String, String>> = OnceCell::new();
static CHINA_LOCATION: OnceCell<ChinaLocation> = OnceCell::new();
static LOCATION_CHANGES: OnceCell<Vec<LocationChange>> = OnceCell::new();
static CHINESE_IDIOMS: OnceCell<FxHashMap<String, u32>> = OnceCell::new();
static XIEHOUYU: OnceCell<FxHashMap<String, String>> = OnceCell::new();
static WORLD_LOCATION: OnceCell<Vec<WorldLocationRecord>> = OnceCell::new();
static QUANTIFIERS: OnceCell<FxHashMap<String, (u64, f64)>> = OnceCell::new();
static PORNOGRAPHY: OnceCell<FxHashSet<String>> = OnceCell::new();
static CHINESE_WORD_DICT: OnceCell<FxHashMap<String, String>> = OnceCell::new();
static TOWN_VILLAGE: OnceCell<FxHashMap<String, FxHashMap<String, Vec<String>>>> = OnceCell::new();
static CHAR_DICT: OnceCell<FxHashMap<char, CharInfo>> = OnceCell::new();
static PHONE_LOCATION: OnceCell<PhoneLocationDict> = OnceCell::new();
static TELECOM_OPERATOR: OnceCell<FxHashMap<String, String>> = OnceCell::new();
static LANDLINE_AREA_CODE: OnceCell<FxHashMap<String, String>> = OnceCell::new();
static PINYIN_PHRASE: OnceCell<FxHashMap<String, Vec<String>>> = OnceCell::new();
static IDF: OnceCell<FxHashMap<String, f64>> = OnceCell::new();
static SENTIMENT_WORDS: OnceCell<FxHashMap<String, f64>> = OnceCell::new();
static SENTIMENT_EXPAND_WORDS: OnceCell<FxHashMap<String, f64>> = OnceCell::new();
static TOPIC_PROMINENCE: OnceCell<Option<TopicProminence>> = OnceCell::new();

/// Admin code → (province, city, county) mapping extracted from
/// `china_location.zip`. Codes are the 6-digit strings at each of the top
/// three administrative levels (prov / city / county).
///
/// `aliases` carries the short-form nickname (if any) for each admin name:
/// `"广东省" → "广东"`, `"上海市" → "上海"`. Used by `location_parser`.
///
/// `entries` preserves every (code, prov, city, county) row from the
/// source file. Some admin codes (e.g. Hong Kong's 810000) are shared
/// across multiple districts, so `codes` — which dedupes by code — is
/// lossy. `location_parser` iterates `entries` to see every district.
#[derive(Debug)]
pub struct ChinaLocation {
    pub codes: FxHashMap<String, (String, Option<String>, Option<String>)>,
    pub aliases: FxHashMap<String, String>,
    pub entries: Vec<(String, String, Option<String>, Option<String>)>,
}

/// World location entry — `(continent, country, capital, cities)`.
#[derive(Debug, Clone)]
pub struct WorldLocationRecord {
    pub continent: String,
    pub country: String,
    pub full_name: String,
    pub capital: Option<String>,
    pub cities: Vec<String>,
}

/// A single old→new administrative change entry parsed from
/// `china_location_change.txt`. `old_loc` / `new_loc` are `(prov, city,
/// county)` — county may be `None` when the change is at city level only.
#[derive(Debug, Clone)]
pub struct LocationChange {
    pub date: String,
    pub department: String,
    /// Old (full, alias) triples — one per level (prov / city / county).
    pub old_loc: [(Option<String>, Option<String>); 3],
    pub new_loc: (Option<String>, Option<String>, Option<String>),
}

/// Per-character dictionary entry from `chinese_char_dictionary.zip`.
#[derive(Debug, Clone)]
pub struct CharInfo {
    pub radical: String,
    pub structure: &'static str,
    pub corner_coding: String,
    pub stroke_order: String,
    pub traditional_version: String,
    pub wubi_coding: String,
    /// Pinyin readings (standard accented form, e.g. "yī", "zhōng"). Most
    /// chars have exactly one; heteronyms (多音字) may have multiple. The
    /// first entry is the primary reading.
    pub pinyin: Vec<String>,
}

/// Output of `phone_location()` — three separate maps carved out of the same
/// `phone_location.zip`.
#[derive(Debug)]
pub struct PhoneLocationDict {
    /// 7-digit cell-phone prefix → "province city" e.g. "1300000" → "山东 济南".
    pub cell_prefix: FxHashMap<String, String>,
    /// 6-digit zip code → "province city".
    pub zip_code: FxHashMap<String, String>,
    /// Landline area code (3-4 digit) → "province city".
    pub area_code: FxHashMap<String, String>,
}

/// LDA topic-prominence data derived from `topic_word_weight.json`. Used by
/// `extract_keyphrase` / `extract_summary` to weight candidates by
/// "topic-specificness" (high = this word is concentrated in few topics and
/// therefore signals a specific subject; low = generic / spread across all
/// topics).
///
/// `per_word` is a min-max-normalized [0.0, 1.0] score per token. `unk` is
/// the fallback for tokens not in the LDA vocabulary (≈ mean / 2, matching
/// Python's `unk_topic_prominence_value`).
#[derive(Debug)]
pub struct TopicProminence {
    pub per_word: FxHashMap<String, f64>,
    pub unk: f64,
}

/// Initialize the global dictionary root. Idempotent — second call is a no-op.
///
/// `path` should point to a directory containing `stopwords.txt`,
/// `sim2tra_char.txt`, etc.
pub fn init_from_path<P: AsRef<Path>>(path: P) -> Result<()> {
    let p = path.as_ref();
    if !p.is_dir() {
        return Err(Error::InvalidArg(format!(
            "dictionary path does not exist or is not a directory: {}",
            p.display()
        )));
    }
    let _ = DICT_ROOT.set(p.to_path_buf());
    Ok(())
}

/// Initialize using the bundled dictionary that ships with this crate.
///
/// The crate includes a 9 MB `data/` subdirectory (compressed) with all
/// runtime-required resources. This is the recommended entry point for
/// applications that use jionlp-core as a library dependency — no need to
/// know the on-disk layout of the upstream Python project.
///
/// Resolves the path via `CARGO_MANIFEST_DIR` at compile time, so it works
/// both in-workspace and after `cargo install`.
pub fn init_default() -> Result<()> {
    let crate_dir = env!("CARGO_MANIFEST_DIR");
    let dict_dir = std::path::PathBuf::from(crate_dir).join("data");
    init_from_path(&dict_dir)
}

fn dict_root() -> Result<&'static Path> {
    DICT_ROOT
        .get()
        .map(|p| p.as_path())
        .ok_or(Error::DictNotInitialized("DICT_ROOT"))
}

fn read_lines(filename: &str) -> Result<Vec<String>> {
    let path = dict_root()?.join(filename);
    let content = fs::read_to_string(&path).map_err(|e| Error::DictIo {
        path: path.display().to_string(),
        source: e,
    })?;
    Ok(content
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect())
}

/// Extract a single entry (`entry_name`) from a zip file under the dict root
/// and return it decoded as UTF-8.
fn read_zip_entry(zip_name: &str, entry_name: &str) -> Result<String> {
    let path = dict_root()?.join(zip_name);
    let f = fs::File::open(&path).map_err(|e| Error::DictIo {
        path: path.display().to_string(),
        source: e,
    })?;
    let mut archive = zip::ZipArchive::new(f).map_err(|e| Error::DictIo {
        path: path.display().to_string(),
        source: std::io::Error::new(std::io::ErrorKind::InvalidData, format!("zip: {}", e)),
    })?;
    let mut entry = archive.by_name(entry_name).map_err(|e| Error::DictIo {
        path: format!("{}::{}", path.display(), entry_name),
        source: std::io::Error::new(std::io::ErrorKind::NotFound, format!("zip entry: {}", e)),
    })?;
    let mut buf = String::new();
    entry.read_to_string(&mut buf).map_err(|e| Error::DictIo {
        path: format!("{}::{}", path.display(), entry_name),
        source: e,
    })?;
    Ok(buf)
}

/// Load a tab-separated key-value file (e.g. `sim2tra_char.txt`).
/// Lines with no tab are skipped.
fn read_kv_tab(filename: &str) -> Result<FxHashMap<String, String>> {
    let lines = read_lines(filename)?;
    let mut map = FxHashMap::default();
    map.reserve(lines.len());
    for line in lines {
        if let Some((k, v)) = line.split_once('\t') {
            let k = k.trim();
            let v = v.trim();
            if !k.is_empty() && !v.is_empty() {
                map.insert(k.to_string(), v.to_string());
            }
        }
    }
    Ok(map)
}

// ───────────────────────────── Public loaders ────────────────────────────────

pub fn stopwords() -> Result<&'static FxHashSet<String>> {
    STOPWORDS.get_or_try_init(|| {
        let lines = read_lines("stopwords.txt")?;
        Ok(lines.into_iter().collect())
    })
}

pub fn negative_words() -> Result<&'static FxHashSet<String>> {
    NEGATIVE_WORDS.get_or_try_init(|| {
        let lines = read_lines("negative_words.txt")?;
        Ok(lines.into_iter().collect())
    })
}

pub fn tra2sim_char() -> Result<&'static FxHashMap<String, String>> {
    TRA2SIM_CHAR.get_or_try_init(|| read_kv_tab("tra2sim_char.txt"))
}

pub fn sim2tra_char() -> Result<&'static FxHashMap<String, String>> {
    SIM2TRA_CHAR.get_or_try_init(|| read_kv_tab("sim2tra_char.txt"))
}

pub fn tra2sim_word() -> Result<&'static FxHashMap<String, String>> {
    TRA2SIM_WORD.get_or_try_init(|| read_kv_tab("tra2sim_word.txt"))
}

pub fn sim2tra_word() -> Result<&'static FxHashMap<String, String>> {
    SIM2TRA_WORD.get_or_try_init(|| read_kv_tab("sim2tra_word.txt"))
}

// ─────────────────────── zip-backed dictionaries ────────────────────────────

/// Load the mainland China administrative-division tree from
/// `china_location.zip`. Only provincial/municipal/county levels are
/// retained — town/village entries are discarded (they carry no admin code
/// and aren't needed by `parse_id_card`).
pub fn china_location() -> Result<&'static ChinaLocation> {
    CHINA_LOCATION.get_or_try_init(|| {
        let text = read_zip_entry("china_location.zip", "china_location.txt")?;
        Ok(parse_china_location_text(&text))
    })
}

/// Load the Xinhua word dictionary. Format: `word \t gloss`.
/// Returns `{word → gloss}`.
pub fn chinese_word_dictionary() -> Result<&'static FxHashMap<String, String>> {
    CHINESE_WORD_DICT.get_or_try_init(|| {
        let text = read_zip_entry("chinese_word_dictionary.zip", "chinese_word_dictionary.txt")?;
        let mut out: FxHashMap<String, String> = FxHashMap::default();
        for raw in text.lines() {
            let raw = raw.trim_end();
            if raw.is_empty() {
                continue;
            }
            let mut it = raw.splitn(2, '\t');
            let w = it.next().unwrap_or("");
            let gloss = it.next().unwrap_or("");
            if !w.is_empty() {
                out.insert(w.to_string(), gloss.to_string());
            }
        }
        Ok(out)
    })
}

/// Load the 5-level town/village map from `china_location.zip`. Key is the
/// concatenated `"province+city+county"` path; value is `{town → [village]}`.
///
/// Used by `parse_location_full(..., town_village=true)` to resolve 镇/村
/// names within a matched admin triple.
pub fn town_village_map() -> Result<&'static FxHashMap<String, FxHashMap<String, Vec<String>>>> {
    TOWN_VILLAGE.get_or_try_init(|| {
        let text = read_zip_entry("china_location.zip", "china_location.txt")?;
        let mut out: FxHashMap<String, FxHashMap<String, Vec<String>>> = FxHashMap::default();
        let mut cur_prov = String::new();
        let mut cur_city = String::new();
        let mut cur_county = String::new();
        let mut cur_town = String::new();
        for raw in text.lines() {
            if raw.is_empty() {
                continue;
            }
            let level = raw.chars().take_while(|&c| c == '\t').count();
            let rest = raw.trim_start_matches('\t');
            let name = rest.split('\t').next().unwrap_or("").to_string();
            if name.is_empty() {
                continue;
            }
            match level {
                0 => {
                    cur_prov = name;
                    cur_city.clear();
                    cur_county.clear();
                    cur_town.clear();
                }
                1 => {
                    cur_city = name;
                    cur_county.clear();
                    cur_town.clear();
                }
                2 => {
                    cur_county = name;
                    cur_town.clear();
                }
                3 => {
                    cur_town = name.clone();
                    let key = format!("{}{}{}", cur_prov, cur_city, cur_county);
                    out.entry(key).or_default().entry(name).or_default();
                }
                4 => {
                    let key = format!("{}{}{}", cur_prov, cur_city, cur_county);
                    if !cur_town.is_empty() {
                        out.entry(key)
                            .or_default()
                            .entry(cur_town.clone())
                            .or_default()
                            .push(name);
                    }
                }
                _ => { /* deeper levels not present */ }
            }
        }
        Ok(out)
    })
}

/// Load 歇后语 (Chinese rhyming riddles). Format: `front \t back`.
pub fn xiehouyu() -> Result<&'static FxHashMap<String, String>> {
    XIEHOUYU.get_or_try_init(|| {
        let text = read_zip_entry("xiehouyu.zip", "xiehouyu.txt")?;
        let mut out: FxHashMap<String, String> = FxHashMap::default();
        for raw in text.lines() {
            let raw = raw.trim();
            if raw.is_empty() {
                continue;
            }
            let mut it = raw.splitn(2, '\t');
            let front = it.next().unwrap_or("");
            let back = it.next().unwrap_or("");
            if !front.is_empty() {
                out.insert(front.to_string(), back.to_string());
            }
        }
        Ok(out)
    })
}

/// Load quantifier stats (`word \t count \t probability`).
pub fn quantifiers() -> Result<&'static FxHashMap<String, (u64, f64)>> {
    QUANTIFIERS.get_or_try_init(|| {
        let path = dict_root()?.join("quantifiers_stat.txt");
        let text = fs::read_to_string(&path)
            .map_err(|e| Error::InvalidArg(format!("read {}: {e}", path.display())))?;
        let mut out: FxHashMap<String, (u64, f64)> = FxHashMap::default();
        for raw in text.lines() {
            let raw = raw.trim();
            if raw.is_empty() {
                continue;
            }
            let parts: Vec<&str> = raw.split('\t').collect();
            if parts.len() < 3 {
                continue;
            }
            let count = parts[1].parse::<u64>().unwrap_or(0);
            let prob = parts[2].parse::<f64>().unwrap_or(0.0);
            out.insert(parts[0].to_string(), (count, prob));
        }
        Ok(out)
    })
}

/// Load the pornography/sensitive word list.
pub fn pornography() -> Result<&'static FxHashSet<String>> {
    PORNOGRAPHY.get_or_try_init(|| {
        let text = read_zip_entry("pornography.zip", "pornography.txt")?;
        let mut out: FxHashSet<String> = FxHashSet::default();
        for raw in text.lines() {
            let raw = raw.trim();
            if !raw.is_empty() {
                out.insert(raw.to_string());
            }
        }
        Ok(out)
    })
}

/// Load the 2-level world-location table (continent/country/capital).
/// Format (Python reference):
///   `<continent>:`
///   `\t<country_short>\t<country_full>\t<capital>\t<city1>/<city2>/…`
pub fn world_location() -> Result<&'static [WorldLocationRecord]> {
    WORLD_LOCATION
        .get_or_try_init(|| {
            let path = dict_root()?.join("world_location.txt");
            let text = fs::read_to_string(&path)
                .map_err(|e| Error::InvalidArg(format!("read {}: {e}", path.display())))?;
            let mut records: Vec<WorldLocationRecord> = Vec::new();
            let mut current_continent = String::new();
            for raw in text.lines() {
                if raw.is_empty() {
                    continue;
                }
                // Continent lines end in ":" (e.g. "亚洲:"). Country lines
                // are tab-separated fields without a trailing colon.
                if raw.ends_with(':') {
                    current_continent = raw.trim_end_matches(':').trim().to_string();
                    continue;
                }
                let rest = raw.trim_start_matches('\t');
                let parts: Vec<&str> = rest.split('\t').collect();
                if parts.len() < 2 {
                    continue;
                }
                let country = parts[0].to_string();
                let full_name = parts.get(1).copied().unwrap_or("").to_string();
                let capital = parts
                    .get(2)
                    .copied()
                    .filter(|s| !s.is_empty())
                    .map(String::from);
                let cities = parts
                    .get(3)
                    .copied()
                    .unwrap_or("")
                    .split('/')
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .collect();
                records.push(WorldLocationRecord {
                    continent: current_continent.clone(),
                    country,
                    full_name,
                    capital,
                    cities,
                });
            }
            Ok(records)
        })
        .map(|v| v.as_slice())
}

/// Load the Chinese idiom dictionary. Returns `{idiom → frequency}` where
/// the frequency is a small integer from `chinese_idiom.txt` (used for
/// sampling 成语接龙 by popularity).
pub fn chinese_idioms() -> Result<&'static FxHashMap<String, u32>> {
    CHINESE_IDIOMS.get_or_try_init(|| {
        let text = read_zip_entry("chinese_idiom.zip", "chinese_idiom.txt")?;
        let mut out: FxHashMap<String, u32> = FxHashMap::default();
        for raw in text.lines() {
            let raw = raw.trim();
            if raw.is_empty() {
                continue;
            }
            let mut it = raw.splitn(2, '\t');
            let idiom = it.next().unwrap_or("");
            let freq_str = it.next().unwrap_or("1");
            if idiom.is_empty() {
                continue;
            }
            let freq = freq_str.parse::<u32>().unwrap_or(1);
            out.insert(idiom.to_string(), freq);
        }
        Ok(out)
    })
}

/// Load the old→new administrative-name change log from
/// `china_location_change.txt`. Format:
///   `date \t dept \t prov_full \t prov_alias \t city_full \t city_alias
///    [\t county_full \t county_alias] => new_prov \t new_city [\t new_county]`
pub fn china_location_changes() -> Result<&'static [LocationChange]> {
    LOCATION_CHANGES
        .get_or_try_init(|| {
            let path = dict_root()?.join("china_location_change.txt");
            let text = fs::read_to_string(&path)
                .map_err(|e| Error::InvalidArg(format!("read {}: {e}", path.display())))?;
            let mut out = Vec::new();
            for raw in text.lines() {
                let raw = raw.trim();
                if raw.is_empty() {
                    continue;
                }
                let mut halves = raw.split("=>");
                let old = halves.next().unwrap_or("");
                let new = halves.next().unwrap_or("");
                let o: Vec<&str> = old.split('\t').collect();
                let n: Vec<&str> = new.split('\t').collect();
                // County change: old has 8 fields, new has 3.
                if o.len() == 8 && n.len() >= 3 {
                    out.push(LocationChange {
                        date: o[0].to_string(),
                        department: o[1].to_string(),
                        old_loc: [
                            (Some(o[2].to_string()), Some(o[3].to_string())),
                            (Some(o[4].to_string()), Some(o[5].to_string())),
                            (Some(o[6].to_string()), Some(o[7].to_string())),
                        ],
                        new_loc: (
                            Some(n[0].to_string()),
                            Some(n[1].to_string()),
                            Some(n[2].to_string()),
                        ),
                    });
                } else if o.len() == 6 && n.len() >= 2 {
                    // City-level change (襄樊 → 襄阳).
                    out.push(LocationChange {
                        date: o[0].to_string(),
                        department: o[1].to_string(),
                        old_loc: [
                            (Some(o[2].to_string()), Some(o[3].to_string())),
                            (Some(o[4].to_string()), Some(o[5].to_string())),
                            (None, None),
                        ],
                        new_loc: (Some(n[0].to_string()), Some(n[1].to_string()), None),
                    });
                }
            }
            Ok(out)
        })
        .map(|v| v.as_slice())
}

fn parse_china_location_text(text: &str) -> ChinaLocation {
    let mut codes: FxHashMap<String, (String, Option<String>, Option<String>)> =
        FxHashMap::default();
    let mut aliases: FxHashMap<String, String> = FxHashMap::default();
    let mut entries: Vec<(String, String, Option<String>, Option<String>)> = Vec::new();
    let mut cur_prov: Option<String> = None;
    let mut cur_city: Option<String> = None;

    for raw in text.lines() {
        if raw.is_empty() {
            continue;
        }
        // Count leading tabs to determine nesting level.
        let level = raw.chars().take_while(|&c| c == '\t').count();
        let rest = raw.trim_start_matches('\t');
        let parts: Vec<&str> = rest.split('\t').collect();
        if parts.is_empty() || parts[0].is_empty() {
            continue;
        }

        match level {
            0 => {
                // province: name \t code \t [alias]
                let name = parts[0].to_string();
                cur_prov = Some(name.clone());
                cur_city = None;
                if parts.len() >= 2 {
                    let code = parts[1].to_string();
                    codes
                        .entry(code.clone())
                        .or_insert_with(|| (name.clone(), None, None));
                    entries.push((code, name.clone(), None, None));
                    if parts.len() >= 3 && !parts[2].is_empty() {
                        aliases.insert(name, parts[2].to_string());
                    }
                }
            }
            1 => {
                // city: name \t code \t [alias]
                let name = parts[0].to_string();
                cur_city = Some(name.clone());
                if parts.len() >= 2 {
                    if let Some(ref prov) = cur_prov {
                        let code = parts[1].to_string();
                        codes
                            .entry(code.clone())
                            .or_insert_with(|| (prov.clone(), Some(name.clone()), None));
                        entries.push((code, prov.clone(), Some(name.clone()), None));
                    }
                    if parts.len() >= 3 && !parts[2].is_empty() {
                        aliases.insert(name, parts[2].to_string());
                    }
                }
            }
            2 => {
                // county: name \t code \t [alias]
                let name = parts[0].to_string();
                if parts.len() >= 2 {
                    if let Some(ref prov) = cur_prov {
                        let code = parts[1].to_string();
                        codes.entry(code.clone()).or_insert_with(|| {
                            (prov.clone(), cur_city.clone(), Some(name.clone()))
                        });
                        entries.push((code, prov.clone(), cur_city.clone(), Some(name.clone())));
                    }
                    if parts.len() >= 3 && !parts[2].is_empty() {
                        aliases.insert(name, parts[2].to_string());
                    }
                }
            }
            _ => { /* town/village levels ignored (detail=False mode) */ }
        }
    }

    ChinaLocation {
        codes,
        aliases,
        entries,
    }
}

/// Pull `[pinyin]` tokens out of a meaning/gloss free-text string.
/// Mirrors the Python `pinyin_ptn = re.compile(r'\[[a-zàáāǎòóōǒèéēěìíīǐùúūǔǜǘǖǚǹńňüḿ]{1,8}\]')`.
/// Returns readings in the order they first appear, deduplicated.
fn extract_pinyin_list(tail: &str) -> Vec<String> {
    use once_cell::sync::Lazy;
    use regex::Regex;
    static PY: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"\[([a-zàáāǎòóōǒèéēěìíīǐùúūǔǜǘǖǚǹńňüḿ]{1,8})\]").unwrap());
    let mut seen: FxHashSet<String> = FxHashSet::default();
    let mut out: Vec<String> = Vec::new();
    for cap in PY.captures_iter(tail) {
        let p = cap.get(1).unwrap().as_str().to_string();
        if seen.insert(p.clone()) {
            out.push(p);
        }
    }
    out
}

/// Structure code 0-9 → descriptive name. See
/// `jionlp.dictionary.STRUCTURE_DICT`.
fn structure_name(code: usize) -> &'static str {
    match code {
        0 => "一体结构",
        1 => "左右结构",
        2 => "上下结构",
        3 => "左中右结构",
        4 => "上中下结构",
        5 => "右上包围结构",
        6 => "左上包围结构",
        7 => "左下包围结构",
        8 => "全包围结构",
        9 => "半包围结构",
        _ => "一体结构",
    }
}

/// Parse the `phone_location.txt` tree: lines without leading tab are
/// `location \t area_code \t zip_code`; lines starting with `\t` are
/// `\t first3 \t comma-separated list of XXXX or XXXX-YYYY ranges`, which
/// expand to 4-digit suffixes and together form 7-digit cell-phone prefixes.
pub fn phone_location() -> Result<&'static PhoneLocationDict> {
    PHONE_LOCATION.get_or_try_init(|| {
        let text = read_zip_entry("phone_location.zip", "phone_location.txt")?;
        Ok(parse_phone_location_text(&text))
    })
}

fn parse_phone_location_text(text: &str) -> PhoneLocationDict {
    let mut cell_prefix: FxHashMap<String, String> = FxHashMap::default();
    let mut zip_code: FxHashMap<String, String> = FxHashMap::default();
    let mut area_code: FxHashMap<String, String> = FxHashMap::default();

    let mut cur_location = String::new();

    for raw in text.lines() {
        if raw.is_empty() {
            continue;
        }
        if let Some(body) = raw.strip_prefix('\t') {
            // Range line: `first3 \t csv`.
            let mut it = body.split('\t');
            let first3 = match it.next() {
                Some(s) => s.trim(),
                None => continue,
            };
            let csv = match it.next() {
                Some(s) => s.trim(),
                None => continue,
            };
            if cur_location.is_empty() {
                continue;
            }
            for part in csv.split(',') {
                let part = part.trim();
                if let Some((start, end)) = part.split_once('-') {
                    let s: u32 = match start.parse() {
                        Ok(n) => n,
                        Err(_) => continue,
                    };
                    let e: u32 = match end.parse() {
                        Ok(n) => n,
                        Err(_) => continue,
                    };
                    for n in s..=e {
                        cell_prefix.insert(format!("{}{:04}", first3, n), cur_location.clone());
                    }
                } else if !part.is_empty() {
                    cell_prefix.insert(format!("{}{}", first3, part), cur_location.clone());
                }
            }
        } else {
            // Header line: `location \t area_code \t zip_code`.
            let parts: Vec<&str> = raw.trim().split('\t').collect();
            if parts.len() >= 3 {
                cur_location = parts[0].to_string();
                area_code.insert(parts[1].to_string(), cur_location.clone());
                zip_code.insert(parts[2].to_string(), cur_location.clone());
            }
        }
    }

    PhoneLocationDict {
        cell_prefix,
        zip_code,
        area_code,
    }
}

/// Load `telecom_operator.txt`: 3-digit prefix → "中国移动" / "中国联通" / ...
pub fn telecom_operator() -> Result<&'static FxHashMap<String, String>> {
    TELECOM_OPERATOR.get_or_try_init(|| {
        let lines = read_lines("telecom_operator.txt")?;
        let mut m = FxHashMap::default();
        for line in lines {
            // Format: "130 中国联通" (space-separated).
            if let Some((k, v)) = line.split_once(char::is_whitespace) {
                m.insert(k.trim().to_string(), v.trim().to_string());
            }
        }
        Ok(m)
    })
}

/// Load `landline_phone_area_code.txt`: area_code → "province/city".
pub fn landline_area_code() -> Result<&'static FxHashMap<String, String>> {
    LANDLINE_AREA_CODE.get_or_try_init(|| {
        let lines = read_lines("landline_phone_area_code.txt")?;
        let mut m = FxHashMap::default();
        for line in lines {
            if let Some((k, v)) = line.split_once(char::is_whitespace) {
                m.insert(k.trim().to_string(), v.trim().to_string());
            }
        }
        Ok(m)
    })
}

/// Load `pinyin_phrase.zip`: phrase → Vec of pinyin readings (one per char
/// of the phrase). Values are slash-separated in the source file; `<py_unk>`
/// entries mean "reading unavailable for this char".
pub fn pinyin_phrase() -> Result<&'static FxHashMap<String, Vec<String>>> {
    PINYIN_PHRASE.get_or_try_init(|| {
        let text = read_zip_entry("pinyin_phrase.zip", "pinyin_phrase.txt")?;
        let mut map: FxHashMap<String, Vec<String>> = FxHashMap::default();
        for raw in text.lines() {
            if raw.is_empty() {
                continue;
            }
            let (key, val) = match raw.split_once('\t') {
                Some(p) => p,
                None => continue,
            };
            let key = key.trim();
            if key.is_empty() {
                continue;
            }
            let readings: Vec<String> = val
                .split('/')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if !readings.is_empty() {
                map.insert(key.to_string(), readings);
            }
        }
        Ok(map)
    })
}

/// Load `sentiment_words.zip`: word → signed score (typically -1.9 .. +1.9).
pub fn sentiment_words() -> Result<&'static FxHashMap<String, f64>> {
    SENTIMENT_WORDS.get_or_try_init(|| {
        let text = read_zip_entry("sentiment_words.zip", "sentiment_words.txt")?;
        let mut map = FxHashMap::default();
        for raw in text.lines() {
            if raw.is_empty() {
                continue;
            }
            if let Some((k, v)) = raw.split_once('\t') {
                if let Ok(score) = v.trim().parse::<f64>() {
                    map.insert(k.trim().to_string(), score);
                }
            }
        }
        Ok(map)
    })
}

/// Load `sentiment_expand_words.txt`: adverb → multiplier (e.g. 非常 → 1.6).
pub fn sentiment_expand_words() -> Result<&'static FxHashMap<String, f64>> {
    SENTIMENT_EXPAND_WORDS.get_or_try_init(|| {
        let lines = read_lines("sentiment_expand_words.txt")?;
        let mut map = FxHashMap::default();
        for line in lines {
            if let Some((k, v)) = line.split_once('\t') {
                if let Ok(score) = v.trim().parse::<f64>() {
                    map.insert(k.trim().to_string(), score);
                }
            }
        }
        Ok(map)
    })
}

/// Load the IDF-weighted dictionary from `idf.zip`. Lines are
/// `<term>\t<idf_value>`, where higher IDF = rarer term = more distinctive.
pub fn idf() -> Result<&'static FxHashMap<String, f64>> {
    IDF.get_or_try_init(|| {
        let text = read_zip_entry("idf.zip", "idf.txt")?;
        let mut map: FxHashMap<String, f64> = FxHashMap::default();
        for raw in text.lines() {
            if raw.is_empty() {
                continue;
            }
            if let Some((k, v)) = raw.split_once('\t') {
                if let Ok(score) = v.trim().parse::<f64>() {
                    map.insert(k.trim().to_string(), score);
                }
            }
        }
        Ok(map)
    })
}

/// Load the LDA topic-prominence table. Computed once from
/// `topic_word_weight.json`, which is **not bundled in the crates.io
/// tarball** (too large — see the note in `Cargo.toml`'s `include` field).
///
/// Returns `Ok(None)` when the file is missing — this is expected for Rust
/// consumers who installed from crates.io without fetching the full
/// `jionlp-data-v*.tar.gz`. Callers (summary / keyphrase) use this to
/// decide whether to apply the topic-weight multiplier.
///
/// Algorithm (matches Python `extract_summary._topic_prominence`):
///   1. For each word w: compute KL(P(topic|w) ‖ uniform) as
///      `sum_i p_i * log2(p_i * N)`, where `p_i = P(topic=i | w)` (fallback
///      `1e-5` for topics the word doesn't appear in) and N is the topic
///      count.
///   2. Min-max normalize across all words to [0, 1].
///   3. `unk = mean / 2` — fallback for OOV words.
pub fn topic_prominence() -> Result<Option<&'static TopicProminence>> {
    let cell = TOPIC_PROMINENCE.get_or_try_init(|| -> Result<Option<TopicProminence>> {
        let root = dict_root()?;
        let path = root.join("topic_word_weight.zip");
        if !path.exists() {
            return Ok(None);
        }
        let text = read_zip_entry("topic_word_weight.zip", "topic_word_weight.json")?;
        let raw: FxHashMap<String, FxHashMap<String, f64>> =
            serde_json::from_str(&text).map_err(|e| Error::DictIo {
                path: path.display().to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("json: {}", e),
                ),
            })?;

        // Topic count = 1 + max topic id across all entries. LDA files
        // store topics as stringified ints ("0".."N-1").
        let mut max_topic: usize = 0;
        for inner in raw.values() {
            for k in inner.keys() {
                if let Ok(tid) = k.parse::<usize>() {
                    if tid > max_topic {
                        max_topic = tid;
                    }
                }
            }
        }
        let num_topics = max_topic + 1;
        if num_topics == 0 || raw.is_empty() {
            return Ok(Some(TopicProminence {
                per_word: FxHashMap::default(),
                unk: 0.0,
            }));
        }

        let mut kl_by_word: FxHashMap<String, f64> = FxHashMap::default();
        kl_by_word.reserve(raw.len());
        let n_f = num_topics as f64;
        for (word, topics) in &raw {
            let mut kl = 0.0_f64;
            for i in 0..num_topics {
                let p = topics.get(&i.to_string()).copied().unwrap_or(1e-5);
                if p > 0.0 {
                    kl += p * (p * n_f).log2();
                }
            }
            kl_by_word.insert(word.clone(), kl);
        }

        // Min-max normalize to [0, 1].
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &v in kl_by_word.values() {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        let span = hi - lo;
        if span > 0.0 {
            for v in kl_by_word.values_mut() {
                *v = (*v - lo) / span;
            }
        }

        let sum: f64 = kl_by_word.values().sum();
        let unk = if kl_by_word.is_empty() {
            0.0
        } else {
            sum / (2.0 * kl_by_word.len() as f64)
        };

        Ok(Some(TopicProminence {
            per_word: kl_by_word,
            unk,
        }))
    })?;
    Ok(cell.as_ref())
}

/// Load the per-character dictionary from `chinese_char_dictionary.zip`.
/// Only the structural fields (radical / structure / codings / strokes /
/// traditional variant) are retained; the pinyin+gloss free-text tail is
/// dropped to keep the map compact.
pub fn char_dictionary() -> Result<&'static FxHashMap<char, CharInfo>> {
    CHAR_DICT.get_or_try_init(|| {
        let text = read_zip_entry("chinese_char_dictionary.zip", "chinese_char_dictionary.txt")?;
        let mut map: FxHashMap<char, CharInfo> = FxHashMap::default();
        for raw in text.lines() {
            if raw.is_empty() {
                continue;
            }
            let parts: Vec<&str> = raw.split('\t').collect();
            if parts.len() < 7 {
                continue;
            }
            let ch = match parts[0].chars().next() {
                Some(c) => c,
                None => continue,
            };
            let structure_code: usize = parts[2].parse().unwrap_or(0);
            // parts[7..] is the pinyin+meaning free-text tail (re-joined with
            // tabs in case the raw had tabs). We scan for `[...]` tokens and
            // keep each one as a pinyin reading.
            let tail: String = parts[7..].join("\t");
            let pinyin = extract_pinyin_list(&tail);
            map.insert(
                ch,
                CharInfo {
                    radical: parts[1].to_string(),
                    structure: structure_name(structure_code),
                    corner_coding: parts[3].to_string(),
                    stroke_order: parts[4].to_string(),
                    traditional_version: parts[5].to_string(),
                    wubi_coding: parts[6].to_string(),
                    pinyin,
                },
            );
        }
        Ok(map)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;

    static INIT: Once = Once::new();

    pub(crate) fn ensure_init() {
        INIT.call_once(|| {
            init_default().expect("init_default");
        });
    }

    #[test]
    fn loads_stopwords() {
        ensure_init();
        let sw = stopwords().unwrap();
        assert!(
            sw.len() > 100,
            "stopwords should be non-trivial, got {}",
            sw.len()
        );
    }

    #[test]
    fn loads_tra2sim_char() {
        ensure_init();
        let m = tra2sim_char().unwrap();
        // "獃" → "呆" (from tra2sim_char.txt line 1)
        assert!(m.len() > 1000);
        assert_eq!(m.get("獃").map(String::as_str), Some("呆"));
    }

    #[test]
    fn loads_sim2tra_char() {
        ensure_init();
        let m = sim2tra_char().unwrap();
        assert!(m.len() > 1000);
    }

    #[test]
    fn loads_china_location_zip() {
        ensure_init();
        let cl = china_location().unwrap();
        // Sanity: at least 3000 codes across prov/city/county levels.
        assert!(cl.codes.len() > 3000, "codes={}", cl.codes.len());

        // 440000 is the Guangdong provincial code (no self-overlap).
        let (prov, city, county) = cl.codes.get("440000").expect("440000");
        assert_eq!(prov, "广东省");
        assert_eq!(city, &None);
        assert_eq!(county, &None);

        // 440100 is Guangzhou city under Guangdong.
        let (prov, city, _) = cl.codes.get("440100").expect("440100");
        assert_eq!(prov, "广东省");
        assert_eq!(city.as_deref(), Some("广州市"));
    }

    #[test]
    fn loads_char_dictionary_zip() {
        ensure_init();
        let m = char_dictionary().unwrap();
        assert!(m.len() > 5000, "char dict size = {}", m.len());
        // 一 is the most common Chinese char.
        let one = m.get(&'一').expect("一 should exist");
        assert_eq!(one.structure, "一体结构");
        assert!(!one.radical.is_empty());
    }

    #[test]
    fn loads_phone_location_zip() {
        ensure_init();
        let pl = phone_location().unwrap();
        // Tens of thousands of 7-digit cell prefixes.
        assert!(
            pl.cell_prefix.len() > 10_000,
            "cell prefixes = {}",
            pl.cell_prefix.len()
        );
        assert!(
            pl.area_code.len() > 300,
            "area codes = {}",
            pl.area_code.len()
        );
        // Area code 010 → 北京/北京.
        assert!(pl.area_code.get("010").unwrap().contains("北京"));
    }

    #[test]
    fn loads_telecom_operator_txt() {
        ensure_init();
        let op = telecom_operator().unwrap();
        assert!(op.len() >= 20, "ops = {}", op.len());
        assert_eq!(op.get("138").map(String::as_str), Some("中国移动"));
        assert_eq!(op.get("130").map(String::as_str), Some("中国联通"));
    }

    #[test]
    fn loads_landline_area_code_txt() {
        ensure_init();
        let lc = landline_area_code().unwrap();
        assert!(lc.len() > 300);
        assert!(lc.get("010").unwrap().contains("北京"));
    }

    #[test]
    fn loads_pinyin_phrase_zip() {
        ensure_init();
        let pp = pinyin_phrase().unwrap();
        assert!(pp.len() > 10_000, "pinyin phrases = {}", pp.len());
        // "一丁不识" → ["yī", "dīng", "bù", "shí"]
        let v = pp.get("一丁不识").unwrap();
        assert_eq!(v, &vec!["yī", "dīng", "bù", "shí"]);
    }

    #[test]
    fn loads_idf_zip() {
        ensure_init();
        let idf = idf().unwrap();
        assert!(idf.len() > 10_000, "idf entries = {}", idf.len());
        // "的" has the lowest IDF in the corpus.
        assert!(idf.get("的").unwrap() < &1.0);
    }
}
