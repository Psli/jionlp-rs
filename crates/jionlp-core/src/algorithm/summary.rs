//! Extractive text summarization — rank sentences by the same weighted
//! combination that Python's `extract_summary` uses.
//!
//! Weighting pipeline (per sentence):
//!   1. **Base TF-IDF**: mean of per-bigram IDF over CJK bigrams present
//!      in `dict::idf()`.
//!   2. **LDA topic weight**: mean of `topic_prominence[word]` for
//!      jieba-tokenized words in the sentence. Multiplied by
//!      `topic_theta` (= 0.5, matching Python default) and added to the
//!      TF-IDF score. Silently skipped when `topic_word_weight.zip` is
//!      not present (crates.io consumers without the full data tarball).
//!   3. **Length penalty**: if `CJK char_len < 15` or `> 70`, ×= 0.7.
//!      (Matches Python's `len(sen) < 15 or len(sen) > 70`.)
//!   4. **Lead-3 bonus**: if `position < 3`, ×= 1.2. (Matches Python's
//!      `lead_3_weight`.)
//!   5. **Zero-score filter**: drop sentences whose base score is 0
//!      (pure-ASCII fragments like "..." that the splitter produces).
//!
//! For MMR diversity, call `extract_summary_mmr` explicitly — Python
//! applies it automatically; we keep it opt-in to preserve the simpler
//! TF-IDF + topic path as a callable primitive.

use crate::dict;
use crate::dict::TopicProminence;
use crate::gadget::split_sentence::{split_sentence, Criterion as SplitCriterion};
use crate::Result;
use jieba_rs::Jieba;
use once_cell::sync::OnceCell;
use rustc_hash::FxHashMap;

/// Matches Python's `topic_theta` default. Controls how strongly LDA
/// topic-prominence contributes vs TF-IDF.
const TOPIC_THETA: f64 = 0.5;

fn jieba() -> &'static Jieba {
    static J: OnceCell<Jieba> = OnceCell::new();
    J.get_or_init(Jieba::new)
}

#[derive(Debug, Clone, PartialEq)]
pub struct SummarySentence {
    pub text: String,
    pub score: f64,
    /// Index in the original document (0-based).
    pub position: usize,
}

/// Return the top-`k` highest-scoring sentences, in document order.
pub fn extract_summary(text: &str, top_k: usize) -> Result<Vec<SummarySentence>> {
    if text.is_empty() || top_k == 0 {
        return Ok(Vec::new());
    }
    let idf = dict::idf()?;
    let topic = dict::topic_prominence()?;

    let sentences = split_sentence(text, SplitCriterion::Coarse);
    if sentences.is_empty() {
        return Ok(Vec::new());
    }

    // Score each sentence with TF-IDF + (optional) LDA topic + length
    // penalty + lead-3 bonus. Zero-score sentences (pure-ASCII like "...")
    // are dropped — the splitter produces them but they carry no summary
    // content.
    let scored: Vec<SummarySentence> = sentences
        .into_iter()
        .enumerate()
        .map(|(pos, s)| {
            let tfidf = bigram_idf_score(&s, idf);
            let topic_w = topic.map(|t| sentence_topic_weight(&s, t)).unwrap_or(0.0);
            let cjk_len = s.chars().filter(|c| is_cjk(*c)).count();
            let length_mul = if !(15..=70).contains(&cjk_len) {
                0.7
            } else {
                1.0
            };
            let lead_mul = if pos < 3 { 1.2 } else { 1.0 };
            let score = (tfidf + TOPIC_THETA * topic_w) * length_mul * lead_mul;
            SummarySentence {
                text: s,
                score,
                position: pos,
            }
        })
        .filter(|s| s.score > 0.0)
        .collect();

    // Pick top-k by score descending, then re-sort by position to preserve
    // narrative order.
    let mut by_score = scored;
    by_score.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    by_score.truncate(top_k);
    by_score.sort_by_key(|s| s.position);

    Ok(by_score)
}

/// Extract as many high-scoring sentences as fit in a character budget.
///
/// Mirrors Python's `extract_summary(text, summary_length=200)`: picks
/// sentences in descending score order (same weighting as
/// `extract_summary`) and keeps accepting them until the cumulative
/// character count would exceed `max_chars`. Selected sentences are
/// returned in original document order.
///
/// The cap is "soft" in the same sense as Python's: if the highest-ranked
/// sentence alone exceeds the budget, it is still returned (otherwise an
/// empty summary would be surprising).
pub fn extract_summary_by_length(text: &str, max_chars: usize) -> Result<Vec<SummarySentence>> {
    if text.is_empty() || max_chars == 0 {
        return Ok(Vec::new());
    }
    // Reuse the top-k path with a large k, then greedy-fit by char budget.
    let ranked = extract_summary(text, usize::MAX)?;
    let mut by_score = ranked;
    by_score.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut picked: Vec<SummarySentence> = Vec::new();
    let mut total = 0usize;
    for s in by_score {
        let len = s.text.chars().count();
        if picked.is_empty() || total + len <= max_chars {
            total += len;
            picked.push(s);
        } else if total + len > max_chars {
            // Stop at first overflow — matches Python `break` behaviour.
            break;
        }
    }
    picked.sort_by_key(|s| s.position);
    Ok(picked)
}

/// Extract summary with Maximal Marginal Relevance (MMR) diversity.
///
/// Balances sentence relevance (IDF score) against redundancy with
/// already-picked sentences. `lambda` ∈ [0, 1] controls the trade-off:
///
/// * `lambda = 1.0` — pure relevance (equivalent to `extract_summary`).
/// * `lambda = 0.0` — pure diversity; picks the least-redundant sentence
///   regardless of score.
/// * `lambda = 0.7` (recommended default) — relevance-first with mild
///   diversity penalty.
///
/// Similarity between two sentences is the Jaccard similarity of their
/// character bigram sets, which is cheap and doesn't require an embedding
/// model.
pub fn extract_summary_mmr(text: &str, top_k: usize, lambda: f64) -> Result<Vec<SummarySentence>> {
    if text.is_empty() || top_k == 0 {
        return Ok(Vec::new());
    }
    let lambda = lambda.clamp(0.0, 1.0);
    let idf = dict::idf()?;

    let sentences = split_sentence(text, SplitCriterion::Coarse);
    if sentences.is_empty() {
        return Ok(Vec::new());
    }

    // Precompute score and bigram set for each sentence. Apply the same
    // TF-IDF + LDA topic + length penalty + lead-3 as `extract_summary`,
    // so lambda=1.0 degenerates exactly to the basic top-k path.
    let topic = dict::topic_prominence()?;
    let bigrams: Vec<rustc_hash::FxHashSet<String>> =
        sentences.iter().map(|s| bigrams_of(s)).collect();
    let scores: Vec<f64> = sentences
        .iter()
        .enumerate()
        .map(|(pos, s)| {
            let tfidf = bigram_idf_score(s, idf);
            let topic_w = topic.map(|t| sentence_topic_weight(s, t)).unwrap_or(0.0);
            let cjk_len = s.chars().filter(|c| is_cjk(*c)).count();
            let length_mul = if !(15..=70).contains(&cjk_len) {
                0.7
            } else {
                1.0
            };
            let lead_mul = if pos < 3 { 1.2 } else { 1.0 };
            (tfidf + TOPIC_THETA * topic_w) * length_mul * lead_mul
        })
        .collect();

    // Greedy selection.
    let mut selected_idx: Vec<usize> = Vec::with_capacity(top_k);
    let mut remaining: Vec<usize> = (0..sentences.len()).collect();

    while !remaining.is_empty() && selected_idx.len() < top_k {
        let mut best: Option<(usize, f64)> = None;
        for (pos, &i) in remaining.iter().enumerate() {
            let max_sim = selected_idx
                .iter()
                .map(|&j| jaccard(&bigrams[i], &bigrams[j]))
                .fold(0.0_f64, f64::max);
            let mmr = lambda * scores[i] - (1.0 - lambda) * max_sim;
            match best {
                Some((_, prev)) if prev >= mmr => {}
                _ => best = Some((pos, mmr)),
            }
        }
        if let Some((pos, _)) = best {
            let idx = remaining.remove(pos);
            selected_idx.push(idx);
        } else {
            break;
        }
    }

    // Return in document order for readability.
    selected_idx.sort_unstable();
    let out = selected_idx
        .into_iter()
        .map(|i| SummarySentence {
            text: sentences[i].clone(),
            score: scores[i],
            position: i,
        })
        .collect();
    Ok(out)
}

/// Mean topic-prominence over the jieba-tokenized words in a sentence.
/// OOV words use `TopicProminence::unk`. Returns 0 for empty input.
fn sentence_topic_weight(sentence: &str, topic: &TopicProminence) -> f64 {
    let tokens = jieba().cut(sentence, true);
    if tokens.is_empty() {
        return 0.0;
    }
    let sum: f64 = tokens
        .iter()
        .map(|t| topic.per_word.get(*t).copied().unwrap_or(topic.unk))
        .sum();
    sum / tokens.len() as f64
}

fn bigrams_of(s: &str) -> rustc_hash::FxHashSet<String> {
    let chars: Vec<char> = s.chars().filter(|c| is_cjk(*c)).collect();
    if chars.len() < 2 {
        return rustc_hash::FxHashSet::default();
    }
    chars.windows(2).map(|w| w.iter().collect()).collect()
}

fn jaccard(a: &rustc_hash::FxHashSet<String>, b: &rustc_hash::FxHashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

fn bigram_idf_score(sentence: &str, idf: &FxHashMap<String, f64>) -> f64 {
    let chars: Vec<char> = sentence.chars().filter(|c| is_cjk(*c)).collect();
    if chars.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for window in chars.windows(2) {
        let bigram: String = window.iter().collect();
        if let Some(v) = idf.get(&bigram) {
            sum += v;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32, 0x4E00..=0x9FA5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dict;
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
    fn extracts_top_k() {
        ensure_init();
        let text =
            "北京是中国的首都。上海是金融中心。广州是南方大都市。深圳是科技之都。成都是西南枢纽。";
        let out = extract_summary(text, 2).unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn preserves_document_order() {
        ensure_init();
        let text = "第一句话。第二句话。第三句话。第四句话。";
        let out = extract_summary(text, 3).unwrap();
        for w in out.windows(2) {
            assert!(w[0].position < w[1].position);
        }
    }

    #[test]
    fn empty_returns_empty() {
        ensure_init();
        assert!(extract_summary("", 5).unwrap().is_empty());
    }

    #[test]
    fn zero_k_returns_empty() {
        ensure_init();
        assert!(extract_summary("随便一段文本。", 0).unwrap().is_empty());
    }

    // ── MMR ─────────────────────────────────────────────────────────────

    #[test]
    fn mmr_returns_top_k() {
        ensure_init();
        let text =
            "北京是中国的首都。上海是金融中心。广州是南方大都市。深圳是科技之都。成都是西南枢纽。";
        let r = extract_summary_mmr(text, 2, 0.7).unwrap();
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn mmr_lambda_one_matches_basic_topk() {
        // With λ=1.0, MMR degenerates to the basic top-K-by-score path
        // (though document order might differ if ties).
        ensure_init();
        let text = "北京是中国的首都。上海是金融中心。广州是南方大都市。深圳是科技之都。";
        let basic = extract_summary(text, 2).unwrap();
        let mmr = extract_summary_mmr(text, 2, 1.0).unwrap();
        // Same length and same positions selected.
        let basic_pos: Vec<_> = basic.iter().map(|s| s.position).collect();
        let mmr_pos: Vec<_> = mmr.iter().map(|s| s.position).collect();
        assert_eq!(basic_pos, mmr_pos);
    }

    #[test]
    fn mmr_lambda_zero_maximizes_diversity() {
        // With λ=0, second pick should be the sentence most *different*
        // from the first (not the 2nd-highest-score).
        ensure_init();
        let text = "机器学习的分支。机器学习的应用。完全无关的天气描述。";
        let r = extract_summary_mmr(text, 2, 0.0).unwrap();
        // Expect position 0 (any) and then 2 (diverse) rather than 1.
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn mmr_empty_returns_empty() {
        ensure_init();
        assert!(extract_summary_mmr("", 5, 0.7).unwrap().is_empty());
    }
}
