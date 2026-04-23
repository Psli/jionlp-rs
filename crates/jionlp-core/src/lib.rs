//! jionlp-core — Rust port of JioNLP (Chinese NLP toolkit).
//!
//! Pure-Rust implementation with no BEAM dependencies — this crate can be
//! used directly by any Rust program, or wrapped by `jionlp-nif` for use
//! from Elixir via Rustler.
//!
//! Heritage: ported from https://github.com/dongrixinyu/JioNLP (Apache-2.0).

// Stylistic clippy lints disabled for the ported codebase. The parity
// target is the Python source structure, and rewriting for these hints
// tends to obscure the line-by-line correspondence without behavioral
// benefit. Add new allows here rather than sprinkling `#[allow]` in
// individual modules.
#![allow(
    // Tolerate lints named by newer rustc/clippy releases that aren't
    // known to the MSRV (1.75) toolchain. Needed because the allow-list
    // below contains names like `doc_overindented_list_items` that were
    // added in clippy 1.86.
    unknown_lints,
    clippy::collapsible_match,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::field_reassign_with_default,
    clippy::if_same_then_else,
    clippy::manual_strip,
    clippy::needless_range_loop,
    clippy::redundant_closure,
    clippy::redundant_guards,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

pub mod algorithm;
pub mod dict;
pub mod gadget;
pub mod rule;
pub mod textaug;
pub mod trie;
pub mod util;

/// Initialize the bundled dictionary — recommended one-line entry point
/// for library users. Shorthand for `dict::init_default()`.
pub fn init() -> Result<()> {
    dict::init_default()
}

pub use util::{absence, bracket, bracket_absence, start_end, TimeIt};

pub use textaug::swap_char_position::swap_char_position;

pub use algorithm::analyse_ner::{analyse_ner_dataset_split, ClassStat, DatasetStats, SplitResult};
pub use algorithm::bpe::{bpe_decode, bpe_encode};
pub use algorithm::data_correction::{correct_cws_sample, correct_pos_sample};
pub use algorithm::entity_compare::{entity_compare_detailed, EntityDiff};
pub use algorithm::extractors::{extract_money, extract_time, MoneyEntity, TimeEntity};
pub use algorithm::keyphrase::{extract_keyphrase, extract_keyphrase_textrank, KeyPhrase};
pub use algorithm::measure::{compute_f1, F1Report, LabelStats};
pub use algorithm::ner::{LexiconNer, NerEntity};
pub use algorithm::ner_accelerate::{TokenBatchBucket, TokenBreakLongSentence, TokenSplitSentence};
pub use algorithm::ner_tools::{
    analyse_ner_dataset, char2word, collect_dataset_entities, is_person_name, token_batch_bucket,
    token_break_long_sentence, token_split_sentence, word2char, NerDatasetAnalysis,
};
pub use algorithm::new_word::new_word_discovery;
pub use algorithm::sentiment::sentiment_score;
pub use algorithm::simhash::{hamming_distance, simhash, simhash_ngram, simhash_similarity};
pub use algorithm::summary::{
    extract_summary, extract_summary_by_length, extract_summary_mmr, SummarySentence,
};
pub use algorithm::tag_conversion::{cws, ner as ner_convert, pos, Entity, F1};
pub use algorithm::text_classification::{
    analyse_dataset as classification_analyse_dataset, analyse_freq_words, DatasetAnalysis,
};
pub use textaug::homophone_substitution::homophone_substitution;
pub use textaug::random_add_delete::random_add_delete;
pub use textaug::replace_entity::{
    replace_entity, EntityAugmented, EntityReplacement, NamedEntity,
};

pub use gadget::char_radical::{char_radical, RadicalInfo};
pub use gadget::china_location_alias::{
    get_china_city_alias, get_china_county_alias, get_china_province_alias, get_china_town_alias,
};
pub use gadget::id_card_parser::{parse_id_card, IdCardInfo};
pub use gadget::idiom_solitaire::IdiomSolitaireGame;
pub use gadget::location_parser::{parse_location_full, LocationParseResult};
pub use gadget::location_recognizer::{
    parse_location, recognize_location, LocationLevel, LocationMatch, ParsedLocation,
};
pub use gadget::lunar_solar::{lunar_to_solar, solar_to_lunar};
pub use gadget::money_num2char::{char2num, num2char, NumStyle};
pub use gadget::money_parser::{parse_money, parse_money_with_default, MoneyInfo};
pub use gadget::motor_vehicle_licence_plate::{parse_motor_vehicle_licence_plate, PlateInfo};
pub use gadget::phone_location::{
    cell_phone_location, landline_phone_location, phone_location, PhoneInfo,
};
pub use gadget::pinyin::{pinyin, PinyinDetail, PinyinEntry, PinyinFormat};
pub use gadget::remove_stopwords::{remove_stopwords, RemoveOpts};
pub use gadget::rule_mining::{mine_rules, LabelRules};
pub use gadget::split_sentence::{split_sentence, Criterion};
pub use gadget::time_parser::{
    parse_time, parse_time_with_ref, DeltaValue, TimeDelta, TimeInfo, TimePeriodInfo,
};
pub use gadget::time_period_parser::normalize_time_period;
pub use gadget::ts_conversion::{sim2tra, tra2sim, TsMode};
pub use rule::checker::{
    check_all_arabic_num, check_all_chinese_char, check_any_arabic_num, check_any_chinese_char,
    check_cell_phone, check_id_card, check_motor_vehicle_licence_plate,
};
pub use rule::cleaners::{
    clean_text, convert_full2half, extract_wechat_id, remove_email, remove_email_with_prefix,
    remove_exception_char, remove_id_card, remove_ip_address, remove_parentheses,
    remove_phone_number, remove_phone_number_with_prefix, remove_qq, remove_url,
    remove_url_with_prefix, replace_chinese, replace_email, replace_id_card, replace_ip_address,
    replace_parentheses, replace_phone_number, replace_qq, replace_url,
};
pub use rule::extractor::{
    extract_cell_phone, extract_chinese, extract_email, extract_id_card, extract_ip_address,
    extract_landline_phone, extract_motor_vehicle_licence_plate, extract_parentheses,
    extract_phone_number, extract_qq, extract_url, Extracted,
};
pub use rule::html_cleansing::{
    clean_html, extract_meta_info, remove_html_tag, remove_menu_div_tag, remove_redundant_char,
};
pub use trie::LabeledTrie;

/// Library-level error type. Most public APIs return `Result<T, Error>`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("dictionary not initialized: {0}")]
    DictNotInitialized(&'static str),
    #[error("dictionary IO failed ({path}): {source}")]
    DictIo {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid argument: {0}")]
    InvalidArg(String),
}

pub type Result<T> = std::result::Result<T, Error>;
