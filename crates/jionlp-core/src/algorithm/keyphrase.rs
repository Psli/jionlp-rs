//! Keyphrase extraction — TF-IDF scoring over word-level n-grams.
//!
//! Port of `jionlp/algorithm/keyphrase`. The Python original depends on
//! `jiojio` (a CRF-based segmenter by the same author). We use
//! [`jieba-rs`](https://crates.io/crates/jieba-rs) instead, which is the
//! Rust-side de-facto standard and handles OOV proper nouns (金正恩 etc.)
//! via its HMM new-word discovery — critical for keyphrase quality on
//! real-world news text.
//!
//! Pipeline:
//!   1. Split text into "runs" of Chinese characters between punctuation.
//!   2. Run jieba on each run (HMM mode) to get word-level tokens.
//!   3. Enumerate consecutive-token windows of 1..=MAX_TOKENS.
//!   4. Filter: stopword boundaries, char length bounds, all-stopword.
//!   5. Score: IDF(phrase) × TF if phrase is in our idf dict; else the
//!      average per-token IDF (with 5.0 fallback for unseen tokens).
//!   6. Rank, truncate to top_k.
//!
//! The TextRank variant (`extract_keyphrase_textrank`) still uses the
//! same jieba tokenization but scores candidates with PageRank over
//! co-occurrence edges instead of TF-IDF.

use crate::dict;
use crate::Result;
use jieba_rs::Jieba;
use once_cell::sync::OnceCell;
use rustc_hash::{FxHashMap, FxHashSet};

/// Punctuation characters that terminate a candidate phrase.
const PUNCTUATION: &[char] = &[
    '，', '。', '！', '？', '、', '；', '：', '“', '”', '‘', '’', '（', '）', '《', '》', '—', '·',
    ',', '.', '!', '?', ';', ':', '"', '\'', '(', ')', '<', '>', '[', ']', '{', '}', '/', '\\',
    '|', '\t', '\n', '\r', ' ',
];

#[derive(Debug, Clone, PartialEq)]
pub struct KeyPhrase {
    pub phrase: String,
    pub weight: f64,
}

/// Shared jieba instance. `Jieba::new()` loads its ~400 KB built-in
/// dictionary and computes the DAG — don't rebuild this per call.
fn jieba() -> &'static Jieba {
    static J: OnceCell<Jieba> = OnceCell::new();
    J.get_or_init(Jieba::new)
}

/// Extract up to `top_k` keyphrases from `text`. `min_n` / `max_n` bound
/// the candidate's **character** length (inclusive). Scores are TF × IDF
/// with known-phrase IDF if available, else average per-token IDF.
///
/// Returns phrases in descending weight order. Empty input or `top_k == 0`
/// yields an empty vec.
pub fn extract_keyphrase(
    text: &str,
    top_k: usize,
    min_n: usize,
    max_n: usize,
) -> Result<Vec<KeyPhrase>> {
    if text.is_empty() || top_k == 0 {
        return Ok(Vec::new());
    }
    let min_n = min_n.max(1);
    let max_n = max_n.max(min_n);

    let idf = dict::idf()?;
    let stopwords = dict::stopwords()?;
    let topic = dict::topic_prominence()?;
    let j = jieba();

    const MAX_TOKENS: usize = 5;
    // Matches Python's default `topic_theta=0.5` in
    // ChineseKeyPhrasesExtractor — the multiplier on the LDA
    // topic-prominence term added to the TF-IDF score.
    const TOPIC_THETA: f64 = 0.5;

    let runs: Vec<String> = split_into_runs(text);

    let mut freq: FxHashMap<String, u32> = FxHashMap::default();
    for run in &runs {
        let tokens: Vec<&str> = j.cut(run, true);
        if tokens.is_empty() {
            continue;
        }
        for w in 1..=MAX_TOKENS.min(tokens.len()) {
            for window in tokens.windows(w) {
                // Stopword at the boundary breaks the phrase.
                if stopwords.contains(*window.first().unwrap())
                    || stopwords.contains(*window.last().unwrap())
                {
                    continue;
                }
                let phrase: String = window.concat();
                let char_len = phrase.chars().count();
                if char_len < min_n || char_len > max_n {
                    continue;
                }
                if is_all_stopwords(&phrase, stopwords) {
                    continue;
                }
                *freq.entry(phrase).or_insert(0) += 1;
            }
        }
    }

    // Collapse substring artifacts: if candidate X is exactly one char
    // shorter than a longer Y (Y = X+1 chars) containing it, and both
    // share the same TF, every occurrence of X lives inside Y — drop X.
    let freq = collapse_substring_dupes(freq);

    let mut scored: Vec<KeyPhrase> = freq
        .into_iter()
        .map(|(phrase, tf)| {
            let toks: Vec<&str> = j.cut(&phrase, true);
            let base = match idf.get(&phrase) {
                Some(v) => *v,
                None => {
                    if toks.is_empty() {
                        char_idf(&phrase, idf)
                    } else {
                        let sum: f64 = toks.iter().map(|t| char_idf(t, idf)).sum();
                        sum / toks.len() as f64
                    }
                }
            };
            // LDA topic weight: mean of per-word topic-prominence. Silent
            // no-op when the LDA data isn't bundled (Rust users on the
            // crates.io subset).
            let topic_w = match topic {
                Some(t) if !toks.is_empty() => {
                    let sum: f64 = toks
                        .iter()
                        .map(|w| t.per_word.get(*w).copied().unwrap_or(t.unk))
                        .sum();
                    sum / toks.len() as f64
                }
                _ => 0.0,
            };
            KeyPhrase {
                phrase,
                weight: (base + TOPIC_THETA * topic_w) * (tf as f64),
            }
        })
        .collect();

    scored.sort_by(|a, b| {
        b.weight
            .partial_cmp(&a.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(top_k);
    Ok(scored)
}

// ───────────────────────── TextRank variant ────────────────────────────────

/// Extract keyphrases using TextRank — a PageRank-style graph ranking
/// over word co-occurrences (not char n-grams).
///
/// ## Algorithm
///
/// 1. Split text into Chinese-char runs (punctuation breaks them).
/// 2. Tokenize each run with jieba (HMM enabled).
/// 3. Generate word-level n-gram candidates (1..=MAX_TOKENS per run),
///    filtering stopword boundaries and char-length bounds.
/// 4. Build an undirected graph: each candidate is a node; two candidates
///    co-occurring in the same run add +1 to their edge weight.
/// 5. Run PageRank for a small fixed number of iterations (default 20).
/// 6. Return top-k by final score.
pub fn extract_keyphrase_textrank(
    text: &str,
    top_k: usize,
    min_n: usize,
    max_n: usize,
) -> Result<Vec<KeyPhrase>> {
    if text.is_empty() || top_k == 0 {
        return Ok(Vec::new());
    }
    let min_n = min_n.max(1);
    let max_n = max_n.max(min_n);
    let stopwords = dict::stopwords()?;
    let j = jieba();

    const MAX_TOKENS: usize = 5;

    let runs = split_into_runs(text);

    let mut all_candidates: FxHashMap<String, usize> = FxHashMap::default();
    let mut run_candidates: Vec<Vec<usize>> = Vec::new();

    for run in &runs {
        let tokens: Vec<&str> = j.cut(run, true);
        let mut cands_this_run: Vec<usize> = Vec::new();
        for w in 1..=MAX_TOKENS.min(tokens.len()) {
            for window in tokens.windows(w) {
                if stopwords.contains(*window.first().unwrap())
                    || stopwords.contains(*window.last().unwrap())
                {
                    continue;
                }
                let phrase: String = window.concat();
                let char_len = phrase.chars().count();
                if char_len < min_n || char_len > max_n {
                    continue;
                }
                if is_all_stopwords(&phrase, stopwords) {
                    continue;
                }
                let next_id = all_candidates.len();
                let id = *all_candidates.entry(phrase).or_insert(next_id);
                cands_this_run.push(id);
            }
        }
        run_candidates.push(cands_this_run);
    }

    let n_nodes = all_candidates.len();
    if n_nodes == 0 {
        return Ok(Vec::new());
    }

    let mut edges: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); n_nodes];
    for cands in &run_candidates {
        for i in 0..cands.len() {
            for j in (i + 1)..cands.len() {
                if cands[i] == cands[j] {
                    continue;
                }
                *edges[cands[i]].entry(cands[j]).or_insert(0.0) += 1.0;
                *edges[cands[j]].entry(cands[i]).or_insert(0.0) += 1.0;
            }
        }
    }

    let damping = 0.85;
    let iters = 20;
    let mut scores = vec![1.0; n_nodes];
    for _ in 0..iters {
        let mut next = vec![1.0 - damping; n_nodes];
        for i in 0..n_nodes {
            let total_weight: f64 = edges[i].values().sum();
            if total_weight <= 0.0 {
                continue;
            }
            for (&neighbor, &w) in &edges[i] {
                next[neighbor] += damping * scores[i] * w / total_weight;
            }
        }
        scores = next;
    }

    let mut id_to_phrase: Vec<(String, usize)> = Vec::with_capacity(n_nodes);
    id_to_phrase.resize(n_nodes, (String::new(), 0));
    for (p, &id) in &all_candidates {
        id_to_phrase[id] = (p.clone(), id);
    }

    let mut out: Vec<KeyPhrase> = id_to_phrase
        .into_iter()
        .map(|(phrase, id)| KeyPhrase {
            phrase,
            weight: scores[id],
        })
        .collect();
    out.sort_by(|a, b| {
        b.weight
            .partial_cmp(&a.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(top_k);
    Ok(out)
}

// ───────────────────────── helpers ────────────────────────────────────────

fn split_into_runs(text: &str) -> Vec<String> {
    let mut runs: Vec<String> = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        if PUNCTUATION.contains(&c) || !is_cjk(c) {
            if !cur.is_empty() {
                runs.push(std::mem::take(&mut cur));
            }
        } else {
            cur.push(c);
        }
    }
    if !cur.is_empty() {
        runs.push(cur);
    }
    runs
}

fn is_cjk(c: char) -> bool {
    matches!(c, '\u{4E00}'..='\u{9FFF}' | '\u{3400}'..='\u{4DBF}')
}

fn is_all_stopwords(phrase: &str, stopwords: &FxHashSet<String>) -> bool {
    phrase.chars().all(|c| stopwords.contains(&c.to_string()))
}

fn char_idf(term: &str, idf: &FxHashMap<String, f64>) -> f64 {
    if let Some(v) = idf.get(term) {
        return *v;
    }
    // Fallback for unseen single chars / short terms: treat as "moderately
    // rare". Avoid zero so the phrase isn't dragged down to 0.
    5.0
}

/// Remove substring artifacts. Drops X when a Y of exactly (chars(X) + 1)
/// characters contains X and has the same TF — these are the "single-char
/// OOV tail" cases that survive into both short and long candidate windows.
fn collapse_substring_dupes(freq: FxHashMap<String, u32>) -> FxHashMap<String, u32> {
    let entries: Vec<(String, u32)> = freq.iter().map(|(k, v)| (k.clone(), *v)).collect();
    let mut out: FxHashMap<String, u32> = FxHashMap::default();
    for (phrase, tf) in &entries {
        let len = phrase.chars().count();
        let dominated = entries.iter().any(|(longer, l_tf)| {
            let llen = longer.chars().count();
            llen == len + 1 && l_tf == tf && longer.contains(phrase.as_str())
        });
        if !dominated {
            out.insert(phrase.clone(), *tf);
        }
    }
    out
}

// ───────────────────────── tests ──────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::Once;

    static INIT: Once = Once::new();
    fn ensure_init() {
        INIT.call_once(|| {
            let manifest = env!("CARGO_MANIFEST_DIR");
            let d = PathBuf::from(manifest).join("data");
            dict::init_from_path(&d).expect("init");
        });
    }

    #[test]
    fn returns_top_k() {
        ensure_init();
        let text = "机器学习是人工智能的一个分支,研究如何从数据中自动学习规律和模式。\
             机器学习广泛应用于自然语言处理、计算机视觉和推荐系统。";
        let r = extract_keyphrase(text, 5, 2, 10).unwrap();
        assert!(r.len() <= 5);
        assert!(!r.is_empty());
        assert!(
            r.iter().any(|k| k.phrase.contains("机器")),
            "expected a 机器* phrase in top 5, got: {:?}",
            r.iter().map(|k| k.phrase.as_str()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn empty_text() {
        ensure_init();
        let r = extract_keyphrase("", 5, 2, 4).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn punctuation_breaks_candidates() {
        ensure_init();
        let text = "苹果,香蕉";
        let r = extract_keyphrase(text, 5, 2, 2).unwrap();
        for k in &r {
            assert!(
                !k.phrase.contains("果香"),
                "crossed punctuation: {}",
                k.phrase
            );
        }
    }

    #[test]
    fn top_k_zero_returns_empty() {
        ensure_init();
        let r = extract_keyphrase("一些文本内容", 0, 2, 4).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn sorted_descending() {
        ensure_init();
        let r = extract_keyphrase(
            "北京是中国的首都。北京有很多名胜古迹。北京人口众多。",
            10,
            2,
            4,
        )
        .unwrap();
        for w in r.windows(2) {
            assert!(w[0].weight >= w[1].weight);
        }
    }

    #[test]
    fn textrank_returns_top_k() {
        ensure_init();
        let text = "机器学习是人工智能的一个分支,研究如何从数据中自动学习规律和模式。\
             机器学习广泛应用于自然语言处理、计算机视觉和推荐系统。";
        let r = extract_keyphrase_textrank(text, 5, 2, 4).unwrap();
        assert!(r.len() <= 5);
        assert!(!r.is_empty());
    }

    #[test]
    fn textrank_sorted_descending() {
        ensure_init();
        let r = extract_keyphrase_textrank(
            "北京是中国的首都。北京有很多名胜古迹。北京人口众多。",
            10,
            2,
            4,
        )
        .unwrap();
        for w in r.windows(2) {
            assert!(w[0].weight >= w[1].weight);
        }
    }

    #[test]
    fn textrank_empty_text() {
        ensure_init();
        let r = extract_keyphrase_textrank("", 5, 2, 4).unwrap();
        assert!(r.is_empty());
    }
}
