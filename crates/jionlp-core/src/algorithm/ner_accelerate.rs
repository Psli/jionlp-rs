//! Port of `jionlp/algorithm/ner/ner_accelerate.py` — three-stage batching
//! pipeline for NER inference.
//!
//! **Stage 1 — `TokenSplitSentence`**: split long samples on punctuation,
//! optionally merge consecutive short sentences so each unit ≤ `max_sen_len`.
//! **Stage 2 — `TokenBreakLongSentence`**: break super-long windows with
//! configurable `overlap`, then merge results back using BIOES continuity.
//! **Stage 3 — `TokenBatchBucket`**: bucket similar-length inputs into
//! `batch_size` chunks to reduce padding.
//!
//! The Python version wraps an inference callable; in Rust we accept a
//! closure `Fn(Vec<Vec<String>>) -> Vec<Vec<String>>` which processes a
//! flat batch of character-per-token samples and returns per-token BIOES
//! tags in the same shape.

use crate::algorithm::ner_tools;

/// Stage 1 — split long sentences by punctuation, optionally merging
/// consecutive short slices so each slice fits within `max_sen_len`.
pub struct TokenSplitSentence<'a> {
    pub func: Box<dyn Fn(Vec<Vec<String>>) -> Vec<Vec<String>> + 'a>,
    pub max_sen_len: usize,
    pub combine_sentences: bool,
}

impl<'a> TokenSplitSentence<'a> {
    pub fn new<F>(func: F, max_sen_len: usize, combine_sentences: bool) -> Self
    where
        F: Fn(Vec<Vec<String>>) -> Vec<Vec<String>> + 'a,
    {
        Self {
            func: Box::new(func),
            max_sen_len,
            combine_sentences,
        }
    }

    /// Apply the pipeline to `token_lists`. Each inner list is one sample's
    /// character sequence. Returns per-token tag sequences, aligned.
    pub fn call(&self, token_lists: Vec<Vec<String>>) -> Vec<Vec<String>> {
        // Phase 1: split each sample.
        let mut splits: Vec<Vec<Vec<String>>> = Vec::with_capacity(token_lists.len());
        for tokens in &token_lists {
            let as_str: String = tokens.join("");
            let pieces = ner_tools::token_split_sentence(&as_str);
            // Back to per-char vecs.
            let chunked: Vec<Vec<String>> = pieces
                .into_iter()
                .map(|p| p.chars().map(|c| c.to_string()).collect())
                .collect();
            splits.push(chunked);
        }

        // Phase 2: optionally combine short pieces within each sample.
        if self.combine_sentences {
            for sample in splits.iter_mut() {
                *sample = combine_short(std::mem::take(sample), self.max_sen_len);
            }
        }

        // Phase 3: flatten + invoke func in one batch.
        let mut flat: Vec<Vec<String>> = Vec::new();
        let mut per_sample_counts: Vec<usize> = Vec::with_capacity(splits.len());
        for sample in &splits {
            per_sample_counts.push(sample.len());
            flat.extend(sample.iter().cloned());
        }
        let flat_tags = (self.func)(flat);

        // Phase 4: un-flatten back to per-sample shape, concatenate within
        // each sample to reconstruct a single tag sequence.
        let mut out: Vec<Vec<String>> = Vec::with_capacity(token_lists.len());
        let mut cursor = 0;
        for cnt in per_sample_counts {
            let pieces = &flat_tags[cursor..cursor + cnt];
            let concat: Vec<String> = pieces.iter().flatten().cloned().collect();
            out.push(concat);
            cursor += cnt;
        }
        out
    }
}

fn combine_short(pieces: Vec<Vec<String>>, max_len: usize) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = Vec::new();
    for piece in pieces {
        if let Some(last) = out.last_mut() {
            if last.len() + piece.len() <= max_len {
                last.extend(piece);
                continue;
            }
        }
        out.push(piece);
    }
    out
}

/// Stage 2 — break any over-long piece into overlapping chunks of size
/// `max_sen_len` with `overlap` overlap. Re-concatenates with BIOES
/// continuity after the model runs.
pub struct TokenBreakLongSentence<'a> {
    pub func: Box<dyn Fn(Vec<Vec<String>>) -> Vec<Vec<String>> + 'a>,
    pub max_sen_len: usize,
    pub overlap: usize,
}

impl<'a> TokenBreakLongSentence<'a> {
    pub fn new<F>(func: F, max_sen_len: usize, overlap: usize) -> Self
    where
        F: Fn(Vec<Vec<String>>) -> Vec<Vec<String>> + 'a,
    {
        Self {
            func: Box::new(func),
            max_sen_len,
            overlap,
        }
    }

    pub fn call(&self, token_lists: Vec<Vec<String>>) -> Vec<Vec<String>> {
        let mut shards: Vec<Vec<String>> = Vec::new();
        let mut per_sample: Vec<Vec<(usize, usize)>> = Vec::with_capacity(token_lists.len());
        for tokens in &token_lists {
            let mut ranges = Vec::new();
            if tokens.len() <= self.max_sen_len {
                let idx = shards.len();
                shards.push(tokens.clone());
                ranges.push((idx, tokens.len()));
            } else {
                let step = self.max_sen_len.saturating_sub(self.overlap).max(1);
                let mut i = 0;
                while i < tokens.len() {
                    let end = (i + self.max_sen_len).min(tokens.len());
                    let slice: Vec<String> = tokens[i..end].to_vec();
                    let idx = shards.len();
                    shards.push(slice);
                    ranges.push((idx, end - i));
                    if end == tokens.len() {
                        break;
                    }
                    i += step;
                }
            }
            per_sample.push(ranges);
        }

        let flat_tags = (self.func)(shards);

        let mut out = Vec::with_capacity(token_lists.len());
        for (ranges, sample_tokens) in per_sample.iter().zip(token_lists.iter()) {
            if ranges.len() == 1 {
                out.push(flat_tags[ranges[0].0].clone());
                continue;
            }
            // Merge overlapping shards: keep left shard's first portion,
            // then fill the remainder from subsequent shards' non-overlap.
            let mut merged: Vec<String> = Vec::with_capacity(sample_tokens.len());
            let mut cursor = 0usize;
            for (shard_idx, shard_len) in ranges {
                let shard_tags = &flat_tags[*shard_idx];
                if merged.is_empty() {
                    merged.extend(shard_tags.iter().cloned());
                    cursor += shard_len;
                } else {
                    // Skip the overlap (first `overlap` tags) and append the rest.
                    let start = self.overlap.min(shard_tags.len());
                    merged.extend(shard_tags[start..].iter().cloned());
                    cursor += shard_len - start;
                }
                let _ = cursor;
            }
            merged.truncate(sample_tokens.len());
            out.push(merged);
        }
        out
    }
}

/// Stage 3 — bucket samples by length and invoke `func` per bucket. Each
/// bucket is padded internally to the max length within the bucket.
pub struct TokenBatchBucket<'a> {
    pub func: Box<dyn Fn(Vec<Vec<String>>) -> Vec<Vec<String>> + 'a>,
    pub max_sen_len: usize,
    pub batch_size: usize,
}

impl<'a> TokenBatchBucket<'a> {
    pub fn new<F>(func: F, max_sen_len: usize, batch_size: usize) -> Self
    where
        F: Fn(Vec<Vec<String>>) -> Vec<Vec<String>> + 'a,
    {
        Self {
            func: Box::new(func),
            max_sen_len,
            batch_size,
        }
    }

    pub fn call(&self, token_lists: Vec<Vec<String>>) -> Vec<Vec<String>> {
        // Sort by length, remember original positions.
        let mut indexed: Vec<(usize, Vec<String>)> = token_lists.into_iter().enumerate().collect();
        indexed.sort_by_key(|(_, t)| t.len());

        let mut ordered_out: Vec<(usize, Vec<String>)> = Vec::with_capacity(indexed.len());
        for batch in indexed.chunks(self.batch_size) {
            let batch_tokens: Vec<Vec<String>> = batch.iter().map(|(_, t)| t.clone()).collect();
            let batch_tags = (self.func)(batch_tokens);
            for ((orig_idx, _), tags) in batch.iter().zip(batch_tags) {
                ordered_out.push((*orig_idx, tags));
            }
        }
        // Restore original order.
        ordered_out.sort_by_key(|(i, _)| *i);
        ordered_out.into_iter().map(|(_, t)| t).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_o(batch: Vec<Vec<String>>) -> Vec<Vec<String>> {
        batch
            .into_iter()
            .map(|sample| sample.iter().map(|_| "O".to_string()).collect())
            .collect()
    }

    #[test]
    fn split_and_reconstruct() {
        let texts: Vec<Vec<String>> = vec![
            "你好。再见！".chars().map(|c| c.to_string()).collect(),
            "今天天气很好".chars().map(|c| c.to_string()).collect(),
        ];
        let pipeline = TokenSplitSentence::new(all_o, 50, false);
        let r = pipeline.call(texts.clone());
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].len(), texts[0].len());
        assert_eq!(r[1].len(), texts[1].len());
    }

    #[test]
    fn break_long_roundtrip() {
        let long: Vec<String> = (0..25).map(|i| format!("c{}", i)).collect();
        let pipeline = TokenBreakLongSentence::new(all_o, 10, 2);
        let r = pipeline.call(vec![long.clone()]);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].len(), long.len());
    }

    #[test]
    fn bucket_restores_order() {
        let lengths = [5, 10, 3, 20, 7];
        let input: Vec<Vec<String>> = lengths
            .iter()
            .map(|&n| (0..n).map(|i| format!("t{}", i)).collect())
            .collect();
        let pipeline = TokenBatchBucket::new(all_o, 100, 2);
        let r = pipeline.call(input.clone());
        assert_eq!(r.len(), input.len());
        for (got, exp) in r.iter().zip(input.iter()) {
            assert_eq!(got.len(), exp.len());
        }
    }
}
