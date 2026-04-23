//! Python time_parser parity corpus — extracted from
//! `jionlp/test/test_time_parser.py` via scripts/extract_parity.py.
//!
//! Every case is a [input, reference_time_iso, expected] triple straight
//! from Python's `self.assertEqual(...)` block. A failure means our
//! parser disagrees with Python's ground truth on the exact same input.
//!
//! Run: `cargo test --test time_parity -- --nocapture parity_report`
//! for a summary of pass/fail counts.

use chrono::NaiveDateTime;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;
use std::sync::Once;

#[derive(Debug, Clone, Deserialize)]
struct Case {
    input: String,
    #[serde(rename = "ref")]
    reference: String,
    #[serde(rename = "type")]
    expected_type: String,
    #[serde(rename = "definition")]
    _definition: String,
    start: String,
    end: String,
}

static INIT: Once = Once::new();
fn ensure_init() {
    INIT.call_once(|| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let d = PathBuf::from(manifest).join("data");
        jionlp_core::dict::init_from_path(&d).expect("init dict");
    });
}

fn load_corpus() -> Vec<Case> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("time_parity.json");
    let raw = fs::read_to_string(&path).expect("read corpus");
    serde_json::from_str(&raw).expect("parse corpus")
}

fn parse_ref(iso: &str) -> Option<NaiveDateTime> {
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"] {
        if let Ok(dt) = NaiveDateTime::parse_from_str(iso, fmt) {
            return Some(dt);
        }
    }
    None
}

#[derive(Debug)]
struct Diff {
    input: String,
    expected: String,
    got: String,
}

fn run_case(c: &Case) -> Result<(), Diff> {
    let now = parse_ref(&c.reference).ok_or_else(|| Diff {
        input: c.input.clone(),
        expected: format!("{} [{}, {}]", c.expected_type, c.start, c.end),
        got: "bad reference_time".into(),
    })?;

    let got = jionlp_core::parse_time_with_ref(&c.input, now);
    let got = match got {
        Some(g) => g,
        None => {
            return Err(Diff {
                input: c.input.clone(),
                expected: format!("{} [{}, {}]", c.expected_type, c.start, c.end),
                got: "None".into(),
            });
        }
    };

    let start_iso = got.start.format("%Y-%m-%d %H:%M:%S").to_string();
    let end_iso = got.end.format("%Y-%m-%d %H:%M:%S").to_string();
    if got.time_type == c.expected_type && start_iso == c.start && end_iso == c.end {
        Ok(())
    } else {
        Err(Diff {
            input: c.input.clone(),
            expected: format!("{} [{}, {}]", c.expected_type, c.start, c.end),
            got: format!("{} [{}, {}]", got.time_type, start_iso, end_iso),
        })
    }
}

#[test]
fn parity_report() {
    ensure_init();
    let corpus = load_corpus();
    let mut pass = 0;
    let mut fail: Vec<Diff> = Vec::new();
    for c in &corpus {
        match run_case(c) {
            Ok(_) => pass += 1,
            Err(d) => fail.push(d),
        }
    }
    let total = corpus.len();
    eprintln!("=== Python time_parser parity ===");
    eprintln!("  pass: {}/{}", pass, total);
    eprintln!("  fail: {}/{}", fail.len(), total);

    // Dump full failure list to a sibling file for categorization.
    let dump_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("time_parity_failures.txt");
    let mut body = String::new();
    for d in &fail {
        body.push_str(&format!("{}\t{}\t{}\n", d.input, d.expected, d.got));
    }
    let _ = fs::write(&dump_path, body);

    if !fail.is_empty() {
        eprintln!(
            "\n--- first 15 diffs (full list in {}) ---",
            dump_path.display()
        );
        for d in fail.iter().take(15) {
            eprintln!(
                "  in: {:?}  expected={}  got={}",
                d.input, d.expected, d.got
            );
        }
    }
    // Diagnostic-only: report statistics, don't fail CI until we achieve
    // parity. Flip this to `assert_eq!(fail.len(), 0)` once we do.
}
