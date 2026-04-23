//! Regression test for extract_keyphrase — guards against the char-n-gram
//! degeneration that produced "朝鲜确认" / "鲜确认金" on realistic input
//! before the FMM-based rewrite.

use std::path::PathBuf;

fn init() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let d = PathBuf::from(manifest).join("data");
    let _ = jionlp_core::dict::init_from_path(&d);
}

#[test]
fn rejects_mid_word_character_slices() {
    init();
    let text = "朝鲜确认金正恩出访俄罗斯 将与普京举行会谈。\
                朝中社星期一报道称,朝鲜最高领导人金正恩将于9月造访俄罗斯,\
                并与俄罗斯总统普京举行会谈。";
    let r = jionlp_core::extract_keyphrase(text, 10, 2, 10).unwrap();
    let phrases: Vec<&str> = r.iter().map(|k| k.phrase.as_str()).collect();

    // Must contain at least some of the actual news-worthy words.
    let expected_any_of = ["俄罗斯", "普京", "朝鲜", "金正恩", "朝中社"];
    let hits: Vec<&&str> = expected_any_of
        .iter()
        .filter(|w| phrases.contains(w))
        .collect();
    assert!(
        hits.len() >= 3,
        "expected ≥3 of {:?} in top 10; got {:?}",
        expected_any_of,
        phrases
    );

    // The specific garbage n-grams the previous impl produced must NOT appear.
    let garbage = ["朝鲜确认", "鲜确认金", "确认金正", "恩出访俄", "出访俄罗"];
    for bad in &garbage {
        assert!(
            !phrases.contains(bad),
            "regression: mid-word slice {:?} in output: {:?}",
            bad,
            phrases
        );
    }
}
