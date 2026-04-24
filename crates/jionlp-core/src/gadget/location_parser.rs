//! Full port of `jionlp/gadget/location_parser.py`.
//!
//! Given a Chinese address string, extract the admin triple (province, city,
//! county) plus a `detail` remainder and an assembled `full_location`. The
//! algorithm builds a candidate list from the admin dictionary (considering
//! both full names and 简称 aliases), scores each candidate by match count
//! and text position, applies a series of tiebreakers, and finally assembles
//! the result dict.
//!
//! Companion low-level scanner: [`crate::gadget::location_recognizer`].

use crate::{dict, Result};
use once_cell::sync::OnceCell;
use rustc_hash::FxHashSet;

/// Structured output — shape-equivalent to Python's `parse_location` dict.
/// `town` / `village` populated only when caller passes `town_village=true`
/// and the admin triple has a matching 4/5-level dictionary entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocationParseResult {
    pub province: Option<String>,
    pub city: Option<String>,
    pub county: Option<String>,
    pub detail: String,
    pub full_location: String,
    pub orig_location: String,
    pub town: Option<String>,
    pub village: Option<String>,
}

/// Parse a Chinese address. `change2new` controls whether deprecated names
/// are silently upgraded to the current ones (e.g. `襄樊市 → 襄阳市`).
///
/// When `town_village=true`, the 4th/5th-level dictionary is consulted to
/// populate `result.town` and `result.village` by matching sub-strings of
/// the parsed `detail` against entries registered under the admin triple.
pub fn parse_location_full(
    text: &str,
    town_village: bool,
    change2new: bool,
) -> Result<LocationParseResult> {
    let idx = index()?;
    let candidates = get_candidates(idx, text);
    if candidates.is_empty() {
        return Ok(LocationParseResult {
            province: None,
            city: None,
            county: None,
            detail: text.to_string(),
            full_location: text.to_string(),
            orig_location: text.to_string(),
            town: None,
            village: None,
        });
    }

    let filtered = filter_candidates(candidates, text);
    if filtered.is_empty() {
        return Ok(LocationParseResult {
            province: None,
            city: None,
            county: None,
            detail: text.to_string(),
            full_location: text.to_string(),
            orig_location: text.to_string(),
            town: None,
            village: None,
        });
    }

    let county_dup = county_duplicate_list(&filtered);
    let winner = &filtered[0];
    let mut res = assemble_final(winner, text, &county_dup, change2new, idx);
    if town_village {
        attach_town_village(&mut res);
    }
    Ok(res)
}

/// Populate `res.town` / `res.village` by looking up the town/village map
/// keyed by prov+city+county and finding any of the entries in `res.detail`.
fn attach_town_village(res: &mut LocationParseResult) {
    let Ok(map) = dict::town_village_map() else {
        return;
    };
    let key = format!(
        "{}{}{}",
        res.province.as_deref().unwrap_or(""),
        res.city.as_deref().unwrap_or(""),
        res.county.as_deref().unwrap_or(""),
    );
    let Some(towns) = map.get(&key) else { return };
    for (town, villages) in towns.iter() {
        if res.detail.contains(town.as_str()) {
            res.town = Some(town.clone());
            for v in villages.iter() {
                if res.detail.contains(v.as_str()) {
                    res.village = Some(v.clone());
                    break;
                }
            }
            break;
        }
    }
}

// ───────────────────────── admin map index ────────────────────────────────

/// A single candidate admin entry. Mirrors Python's 5-tuple of
/// `[code, [prov_full, prov_alias], [city_full, city_alias],
///  [county_full, county_alias], is_new]`, augmented on scoring with
/// `count` and `offsets`.
#[derive(Debug, Clone)]
struct AdminEntry {
    #[allow(dead_code)]
    code: String,
    prov: (Option<String>, Option<String>),
    city: (Option<String>, Option<String>),
    county: (Option<String>, Option<String>),
    is_new: bool,
}

struct AdminIndex {
    entries: Vec<AdminEntry>,
    /// Province canonical name → alias (for filter_candidates' reverse lookup).
    _aliases: rustc_hash::FxHashMap<String, String>,
    /// Old-key → new (prov, city, county) mapping, for change2new.
    old2new: rustc_hash::FxHashMap<String, (Option<String>, Option<String>, Option<String>)>,
}

static ADMIN_INDEX: OnceCell<AdminIndex> = OnceCell::new();

fn index() -> Result<&'static AdminIndex> {
    ADMIN_INDEX.get_or_try_init(build_index)
}

fn build_index() -> Result<AdminIndex> {
    let cl = dict::china_location()?;
    let changes = dict::china_location_changes().ok().unwrap_or(&[]);

    let mut entries: Vec<AdminEntry> = Vec::new();

    // Municipalities (city = province name) to skip when building
    // province-only entries so they go through the city path.
    let municipalities: FxHashSet<&'static str> = ["北京", "上海", "天津", "重庆", "香港", "澳门"]
        .into_iter()
        .collect();

    // A level's alias column can be a `/`-separated list (e.g.
    // 巴音郭楞蒙古自治州's alias is "巴州/巴音郭楞"). Split so each alias
    // appears as its own candidate entry.
    fn split_aliases(s: &Option<String>) -> Vec<Option<String>> {
        match s {
            None => vec![None],
            Some(x) if x.is_empty() => vec![None],
            Some(x) if x.contains('/') => x.split('/').map(|p| Some(p.to_string())).collect(),
            Some(x) => vec![Some(x.clone())],
        }
    }

    // Infer level from row shape (which columns are set) rather than the
    // admin code — some codes (e.g. HK's 810000) are shared across an
    // entire SAR, so code-based classification would erase every district.
    for (code, prov, city, county) in cl.entries.iter() {
        let prov_alias_raw = cl.aliases.get(prov).cloned();
        let city_alias_raw = city.as_ref().and_then(|c| cl.aliases.get(c).cloned());
        let county_alias_raw = county.as_ref().and_then(|c| cl.aliases.get(c).cloned());
        let prov_aliases = split_aliases(&prov_alias_raw);
        let city_aliases = split_aliases(&city_alias_raw);
        let county_aliases = split_aliases(&county_alias_raw);

        match (city.is_some(), county.is_some()) {
            (false, false) => {
                let is_muni = prov_alias_raw
                    .as_deref()
                    .map(|a| {
                        a.split('/').any(|p| municipalities.contains(p))
                    })
                    .unwrap_or(false)
                    || municipalities
                        .iter()
                        .any(|m| prov.starts_with(*m) && prov.ends_with(['市', '区']));
                if !is_muni {
                    for pa in &prov_aliases {
                        entries.push(AdminEntry {
                            code: code.clone(),
                            prov: (Some(prov.clone()), pa.clone()),
                            city: (None, None),
                            county: (None, None),
                            is_new: true,
                        });
                    }
                }
            }
            (true, false) => {
                for pa in &prov_aliases {
                    for ca in &city_aliases {
                        entries.push(AdminEntry {
                            code: code.clone(),
                            prov: (Some(prov.clone()), pa.clone()),
                            city: (city.clone(), ca.clone()),
                            county: (None, None),
                            is_new: true,
                        });
                    }
                }
            }
            (true, true) | (false, true) => {
                // Normalize county: when it ends with 经济技术开发区, store
                // the suffix only — mirrors Python's collision-prevention
                // trick so `河北省秦皇岛市经济技术开发区` resolves to
                // county=经济技术开发区 and detail="".
                let normalized_county = county.as_ref().map(|c| {
                    if c.ends_with("经济技术开发区") && c.chars().count() > 6 {
                        "经济技术开发区".to_string()
                    } else {
                        c.clone()
                    }
                });
                for pa in &prov_aliases {
                    for ca in &city_aliases {
                        for cty_a in &county_aliases {
                            entries.push(AdminEntry {
                                code: code.clone(),
                                prov: (Some(prov.clone()), pa.clone()),
                                city: (city.clone(), ca.clone()),
                                county: (normalized_county.clone(), cty_a.clone()),
                                is_new: true,
                            });
                        }
                    }
                }
            }
        }
    }

    // Register old-name entries and the old→new mapping.
    let mut old2new: rustc_hash::FxHashMap<
        String,
        (Option<String>, Option<String>, Option<String>),
    > = rustc_hash::FxHashMap::default();
    for chg in changes {
        let e = AdminEntry {
            code: "000000".to_string(),
            prov: chg.old_loc[0].clone(),
            city: chg.old_loc[1].clone(),
            county: chg.old_loc[2].clone(),
            is_new: false,
        };
        // Build an old-key from the non-None full names.
        let key: String = [&e.prov.0, &e.city.0, &e.county.0]
            .iter()
            .filter_map(|s| s.as_ref().map(|x| x.as_str()))
            .collect();
        old2new.insert(key, chg.new_loc.clone());
        entries.push(e);
    }

    Ok(AdminIndex {
        entries,
        _aliases: cl.aliases.clone(),
        old2new,
    })
}

// ───────────────────────── candidate generation ───────────────────────────

/// Per-level match metadata: `(offset, alias_flag)`. `offset == -1` means no
/// match at this level. `alias_flag == 0` is full-name match, `1` is alias.
type LevelOffsets = [(i64, i8); 3];

#[derive(Debug, Clone)]
struct ScoredCandidate {
    entry: AdminEntry,
    count: i32,
    offsets: LevelOffsets,
}

fn get_candidates(idx: &AdminIndex, text: &str) -> Vec<ScoredCandidate> {
    let mut out = Vec::new();
    let municipalities = ["北京", "上海", "天津", "重庆", "香港", "澳门"];
    for entry in &idx.entries {
        let mut count = 0;
        let mut offsets: LevelOffsets = [(-1, -1); 3];
        let levels = [&entry.prov, &entry.city, &entry.county];
        let mut broken = false;

        for (i, level) in levels.iter().enumerate() {
            let (full, alias) = (&level.0, &level.1);
            let mut matched: Option<(usize, i8)> = None;
            for (alias_idx, name) in [(0i8, full), (1, alias)].iter() {
                if let Some(n) = name.as_ref() {
                    if n.is_empty() {
                        continue;
                    }
                    if let Some(pos) = text.find(n.as_str()) {
                        // Alias-specific exclusion: if the match is an alias
                        // directly followed by `路/大街/街`, skip — e.g.
                        // `重庆路` in "北海市重庆路其仓11号" shouldn't match
                        // city alias 重庆.
                        if *alias_idx == 1 {
                            let after = &text[pos + n.len()..];
                            if after.starts_with("路")
                                || after.starts_with("大街")
                                || after.starts_with("街")
                            {
                                continue;
                            }
                        }
                        matched = Some((pos, *alias_idx));
                        break;
                    }
                }
            }
            if let Some((pos, alias_idx)) = matched {
                count += 1;
                offsets[i] = (pos as i64, alias_idx);

                // Guard: "青海西宁" would match prov=青海省 at 0, then
                // city=海西蒙古族... at 3 bytes / 1 char apart, which is
                // wrong. Convert byte gap to char gap via the preceding-
                // prefix length to detect.
                fn char_gap(text: &str, a: i64, b: i64) -> i64 {
                    let (lo, hi) = if a <= b {
                        (a as usize, b as usize)
                    } else {
                        (b as usize, a as usize)
                    };
                    text[lo..hi].chars().count() as i64
                }
                if i >= 1
                    && offsets[i - 1].0 >= 0
                    && char_gap(text, offsets[i - 1].0, offsets[i].0) == 1
                {
                    count = 0;
                    broken = true;
                    break;
                }
                if i == 2 && offsets[0].0 >= 0 && char_gap(text, offsets[0].0, offsets[2].0) == 1 {
                    count = 0;
                    broken = true;
                    break;
                }
            }
        }
        if broken || count == 0 {
            continue;
        }
        // Municipality sanity: if prov alias is a municipality word AND the
        // prov alias appears in text (which also covers city), decrement.
        if let Some(pa) = entry.prov.1.as_deref() {
            if municipalities.contains(&pa) && text.contains(pa) {
                count -= 1;
            }
        }
        if count > 0 {
            out.push(ScoredCandidate {
                entry: entry.clone(),
                count,
                offsets,
            });
        }
    }
    out
}

fn filter_candidates(mut candidates: Vec<ScoredCandidate>, _text: &str) -> Vec<ScoredCandidate> {
    // Step 2.0 — Python parity: if offsets have duplicates, find the most
    // common offset; if its two items are (full, alias) in that order,
    // drop the candidate. Three-way nested (e.g. 湘/湘潭/湘潭县 at offset 0)
    // falls through because the first two flags are both 1.
    candidates.retain(|c| {
        let positive: Vec<(i64, i8)> = c
            .offsets
            .iter()
            .filter(|x| x.0 >= 0)
            .copied()
            .collect();
        let offset_set: rustc_hash::FxHashSet<i64> = positive.iter().map(|x| x.0).collect();
        if offset_set.len() == positive.len() {
            return true; // no duplicates
        }
        let mut counts: rustc_hash::FxHashMap<i64, i32> = rustc_hash::FxHashMap::default();
        for (p, _) in &positive {
            *counts.entry(*p).or_insert(0) += 1;
        }
        let most_common = counts.iter().max_by_key(|(_, c)| *c).map(|(k, _)| *k);
        let Some(moff) = most_common else {
            return true;
        };
        let dup: Vec<(i64, i8)> = positive.iter().filter(|x| x.0 == moff).copied().collect();
        // Python: `the_same_offset_loc[0][1] == 0 and the_same_offset_loc[1][1] == 1`.
        // Only drop when the FIRST two entries at that offset are (full, alias).
        !(dup.len() >= 2 && dup[0].1 == 0 && dup[1].1 == 1)
    });
    if candidates.is_empty() {
        return candidates;
    }

    // Step 2.1 — max match count wins.
    let max_count = candidates.iter().map(|c| c.count).max().unwrap_or(0);
    candidates.retain(|c| c.count == max_count);
    if candidates.is_empty() {
        return candidates;
    }

    // Step 2.1.5 — if exactly 2 candidates with same offsets, drop the
    // is_new=false (old) entry in favor of the new one.
    if candidates.len() == 2 {
        let o0: Vec<i64> = candidates[0].offsets.iter().map(|x| x.0).collect();
        let o1: Vec<i64> = candidates[1].offsets.iter().map(|x| x.0).collect();
        if o0 == o1 {
            let keep: Vec<ScoredCandidate> = candidates
                .iter()
                .filter(|c| c.entry.is_new)
                .cloned()
                .collect();
            if !keep.is_empty() {
                candidates = keep;
            }
        }
    }
    if candidates.len() == 1 {
        return candidates;
    }

    // Step 2.2 — earliest total offset wins. Tiebreak by keeping items with
    // the minimum offset-sum (among matched levels).
    fn offset_sum(c: &ScoredCandidate) -> i64 {
        c.offsets.iter().map(|x| x.0).sum::<i64>()
    }
    // Additionally enforce prov/city/county must appear in that order when
    // all three levels are matched.
    let municipalities = ["北京", "上海", "天津", "重庆", "香港", "澳门"];
    candidates.retain(|c| {
        if let Some(pa) = c.entry.prov.1.as_deref() {
            if municipalities.contains(&pa) {
                return true;
            }
        }
        let o = &c.offsets;
        if o.iter().all(|x| x.0 >= 0) {
            // `<=` allows nested aliases — e.g. `湘潭县` → prov=湘 @0,
            // city=湘潭 @0, county=湘潭县 @0.
            o[0].0 <= o[1].0 && o[1].0 <= o[2].0
        } else {
            true
        }
    });
    if candidates.is_empty() {
        return candidates;
    }

    candidates.sort_by_key(offset_sum);
    let min_offset = offset_sum(&candidates[0]);
    candidates.retain(|c| offset_sum(c) == min_offset);

    // Step 2.3 — prefer full-name over alias matches.
    // case 1: min over the non-negative alias_flags for the candidate.
    fn min_alias(c: &ScoredCandidate) -> i8 {
        c.offsets
            .iter()
            .filter_map(|x| if x.0 >= 0 { Some(x.1) } else { None })
            .min()
            .unwrap_or(1)
    }
    let min_a = candidates.iter().map(min_alias).min().unwrap_or(1);
    candidates.retain(|c| min_alias(c) == min_a);

    // case 2: sum of alias_flags — lower is better.
    fn sum_alias(c: &ScoredCandidate) -> i32 {
        c.offsets
            .iter()
            .filter_map(|x| if x.0 >= 0 { Some(x.1 as i32) } else { None })
            .sum()
    }
    let min_sum = candidates.iter().map(sum_alias).min().unwrap_or(0);
    candidates.retain(|c| sum_alias(c) == min_sum);

    // Step 2.4 — if min_a==1 (all alias) and only 1 alias-level matched per
    // candidate, prefer higher admin level (earliest non-empty level).
    let max_alias_matched = candidates
        .iter()
        .map(|c| c.offsets.iter().filter(|x| x.0 >= 0).count())
        .max()
        .unwrap_or(0);
    if min_a == 1 && max_alias_matched == 1 {
        candidates.sort_by_key(|c| {
            c.offsets
                .iter()
                .position(|x| x.0 >= 0)
                .unwrap_or(usize::MAX)
        });
    }

    // Step 3.1 — drop old-name entries whose canonical new-name is already
    // in the candidate set.
    // (simplified: if any entry is_new=false and an entry with the same
    // new_loc exists, drop the old one)
    // NB: old2new lookup happens in assemble_final; we just prefer is_new=true here.
    if candidates.iter().any(|c| c.entry.is_new) {
        candidates.retain(|c| c.entry.is_new);
    }

    candidates
}

fn county_duplicate_list(candidates: &[ScoredCandidate]) -> Vec<String> {
    // Per-candidate county name (full or alias per offsets[2].1).
    let per_candidate: Vec<Option<String>> = candidates
        .iter()
        .map(|c| {
            let (cf, ca) = (&c.entry.county.0, &c.entry.county.1);
            match c.offsets[2].1 {
                0 => cf.clone(),
                1 => ca.clone(),
                _ => None,
            }
        })
        .collect();

    let names: Vec<String> = per_candidate.iter().flatten().cloned().collect();

    // Python exception: if all candidates sharing a county name also share
    // the same city, drop it from the dup list — this covers 库尔勒市
    // appearing via both 巴州 and 巴音郭楞 aliases of the same 巴音郭楞蒙古
    // 自治州, where it's not really a collision.
    let mut exception: rustc_hash::FxHashSet<String> = rustc_hash::FxHashSet::default();
    // Python iterates `county_dup_list` (names per candidate), so an empty
    // sub-list can't happen here. Use the set of distinct names.
    let distinct: rustc_hash::FxHashSet<String> = names.iter().cloned().collect();
    for dup in &distinct {
        // Python bug-for-bug: `city_dup_list` is built from ALL candidates,
        // not only those matching `dup`. Keep the same semantics.
        let city_set: rustc_hash::FxHashSet<Option<String>> = candidates
            .iter()
            .map(|c| c.entry.city.0.clone())
            .collect();
        if city_set.len() == 1 {
            exception.insert(dup.clone());
        }
    }

    let mut counts: rustc_hash::FxHashMap<String, i32> = rustc_hash::FxHashMap::default();
    for n in &names {
        *counts.entry(n.clone()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .filter(|(n, c)| *c > 1 && !exception.contains(n))
        .map(|(n, _)| n)
        .collect()
}

fn assemble_final(
    candidate: &ScoredCandidate,
    text: &str,
    county_dup: &[String],
    change2new: bool,
    idx: &AdminIndex,
) -> LocationParseResult {
    // Step 4 — compute detail_idx as the position right after the deepest
    // matched level's name (full or alias).
    let mut detail_idx: usize = 0;
    let mut prov = None;
    let mut city = None;
    let mut county = None;
    let levels = [
        &candidate.entry.prov,
        &candidate.entry.city,
        &candidate.entry.county,
    ];
    let names_by_level = [
        &candidate.entry.prov,
        &candidate.entry.city,
        &candidate.entry.county,
    ];
    for (i, off) in candidate.offsets.iter().enumerate() {
        if off.0 < 0 {
            continue;
        }
        // Name consumed at this level.
        let (full, alias) = names_by_level[i];
        let name = match off.1 {
            0 => full.clone(),
            1 => alias.clone(),
            _ => None,
        };
        let len = name.as_ref().map(|s| s.len()).unwrap_or(0);
        // Python overwrites detail_idx at each matched level, so the
        // deepest-matched level always wins — even if an earlier level
        // matched a later (second-occurrence) position in the text.
        detail_idx = off.0 as usize + len;
        // Populate levels <= i when not in county_dup.
        let name_str = name.unwrap_or_default();
        if !county_dup.contains(&name_str) {
            prov = levels[0].0.clone();
        }
        if i >= 1 && !county_dup.contains(&name_str) {
            city = levels[1].0.clone();
        }
        if i >= 2 {
            if !county_dup.contains(&name_str) {
                county = levels[2].0.clone();
            } else {
                county = match off.1 {
                    0 => levels[2].0.clone(),
                    1 => levels[2].1.clone(),
                    _ => None,
                };
            }
        }
    }

    // Step 5 — old → new mapping.
    if change2new {
        let key: String = [&prov, &city, &county]
            .iter()
            .filter_map(|s| s.as_ref().map(|x| x.as_str()))
            .collect();
        if let Some((np, nc, ncc)) = idx.old2new.get(&key) {
            prov = np.clone();
            city = nc.clone();
            county = ncc.clone();
        }
    }

    // Step 6 — detail slice.
    let detail = if detail_idx >= text.len() {
        String::new()
    } else {
        let mut d = text[detail_idx..].to_string();
        if d.starts_with('县') {
            d = d.trim_start_matches('县').to_string();
        }
        d
    };

    // Step 7 — strip `直辖` placeholders.
    if city.as_deref().map(|s| s.contains("直辖")).unwrap_or(false) {
        city = None;
    }
    if county
        .as_deref()
        .map(|s| s.contains("直辖"))
        .unwrap_or(false)
    {
        county = None;
    }

    // Step 8 — assemble admin_part.
    let municipalities = ["北京", "上海", "天津", "重庆", "香港", "澳门"];
    let mut admin_part = String::new();
    if let Some(ref p) = prov {
        admin_part.push_str(p);
    }
    if let Some(ref c) = city {
        let is_muni = municipalities.iter().any(|m| c.contains(*m));
        if !is_muni {
            admin_part.push_str(c);
        }
    }
    if let Some(ref cc) = county {
        admin_part.push_str(cc);
    }
    let full_location = format!("{}{}", admin_part, detail);

    LocationParseResult {
        province: prov,
        city,
        county,
        detail,
        full_location,
        orig_location: text.to_string(),
        town: None,
        village: None,
    }
}
