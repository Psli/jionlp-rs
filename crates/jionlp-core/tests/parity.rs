//! Python-Rust parity regression suite.
//!
//! The expected outputs below are curated from JioNLP's Python reference —
//! specifically from docstring examples, README examples, and tests. These
//! golden cases document behaviors Rust must preserve. Add a case here
//! whenever you port a module; when behavior intentionally diverges, update
//! the expectation and note the divergence in PLAN.md's risk log.
//!
//! Rationale for keeping these as Rust tests (not a Python-generated file):
//! the Python jionlp pulls jiojio and several system deps that aren't
//! trivially installable in CI. Golden values are cheap; keep them in
//! source.

use jionlp_core as jio;
use std::path::PathBuf;
use std::sync::Once;

static INIT: Once = Once::new();
fn ensure_init() {
    INIT.call_once(|| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let dict = PathBuf::from(manifest).join("data");
        jio::dict::init_from_path(&dict).expect("init");
    });
}

// ────────────────────────────── gadget ────────────────────────────────────

#[test]
fn parity_split_sentence_fine_from_docstring() {
    // jionlp/gadget/split_sentence.py docstring:
    //   text = '中华古汉语，泱泱大国，历史传承的瑰宝。'
    //   jio.split_sentence(text, criterion='fine')
    //   → ['中华古汉语，', '泱泱大国，', '历史传承的瑰宝。']
    let got = jio::split_sentence(
        "中华古汉语，泱泱大国，历史传承的瑰宝。",
        jio::Criterion::Fine,
    );
    assert_eq!(got, vec!["中华古汉语，", "泱泱大国，", "历史传承的瑰宝。"]);
}

#[test]
fn parity_tra2sim_char_from_docstring() {
    // jionlp/gadget/ts_conversion.py docstring:
    //   text = '今天天氣好晴朗，想喫速食麵。妳還在工作嗎？在太空梭上工作嗎？'
    //   jio.tra2sim(text, mode='char')
    //   → '今天天气好晴朗，想吃速食面。你还在工作吗？在太空梭上工作吗？'
    ensure_init();
    let got = jio::tra2sim(
        "今天天氣好晴朗，想喫速食麵。妳還在工作嗎？在太空梭上工作嗎？",
        jio::TsMode::Char,
    )
    .unwrap();
    assert_eq!(
        got,
        "今天天气好晴朗，想吃速食面。你还在工作吗？在太空梭上工作吗？"
    );
}

#[test]
fn parity_tra2sim_word_idiom() {
    // Word-mode should map "速食麵" → "方便面" and "太空梭" → "航天飞机".
    ensure_init();
    let got = jio::tra2sim(
        "今天天氣好晴朗，想喫速食麵。妳還在工作嗎？在太空梭上工作嗎？",
        jio::TsMode::Word,
    )
    .unwrap();
    assert!(got.contains("方便面"), "expected '方便面' in {got}");
    assert!(got.contains("航天飞机"), "expected '航天飞机' in {got}");
}

#[test]
fn parity_parse_id_card_docstring_case() {
    // Python docstring:
    //   text = '52010320171109002X' → {province: '贵州省', city: '贵阳市', county: '云岩区',
    //   birth_year: '2017', birth_month: '11', birth_day: '09', gender: '女',
    //   check_code: 'x'}
    ensure_init();
    let info = jio::parse_id_card("52010320171109002X").unwrap().unwrap();
    assert_eq!(info.province, "贵州省");
    assert_eq!(info.city.as_deref(), Some("贵阳市"));
    assert_eq!(info.county.as_deref(), Some("云岩区"));
    assert_eq!(info.birth_year, "2017");
    assert_eq!(info.birth_month, "11");
    assert_eq!(info.birth_day, "09");
    assert_eq!(info.gender, "女");
    assert_eq!(info.check_code, "x");
}

#[test]
fn parity_parse_plate_docstring_case() {
    // Python docstring:
    //   text = '川A·23047B' → {car_loc: '川A', car_type: 'PEV', car_size: 'big'}
    let info = jio::parse_motor_vehicle_licence_plate("川A·23047B").unwrap();
    assert_eq!(info.car_loc, "川A");
    assert_eq!(info.car_type, "PEV");
    assert_eq!(info.car_size, Some("big"));
}

#[test]
fn parity_num2char_docstring_sim() {
    // Python docstring:
    //   38009 + sim → '三万八千零九'
    assert_eq!(
        jio::num2char("38009", jio::NumStyle::Simplified).unwrap(),
        "三万八千零九"
    );
}

#[test]
fn parity_num2char_docstring_tra() {
    // Python docstring:
    //   1234 + tra → '壹仟贰佰叁拾肆'
    assert_eq!(
        jio::num2char("1234", jio::NumStyle::Traditional).unwrap(),
        "壹仟贰佰叁拾肆"
    );
}

#[test]
fn parity_pinyin_docstring_standard() {
    // Python docstring:
    //   '中华人民共和国。' + standard → ['zhōng','huá','rén','mín','gòng','hé','guó','<unk>']
    // Our sentinel for non-Chinese is '<py_unk>' (matches Python's
    // self.py_unk). The Python example shows '<unk>' in the simplified
    // README snippet but the source code uses '<py_unk>'.
    ensure_init();
    let r = jio::pinyin("中华人民共和国。", jio::PinyinFormat::Standard).unwrap();
    let plain: Vec<String> = r
        .iter()
        .map(|e| match e {
            jio::PinyinEntry::Standard(s) => s.clone(),
            _ => panic!("wrong variant"),
        })
        .collect();
    let joined = plain.join(",");
    // Dictionary gives first reading; verify the content chars look right.
    assert!(plain.iter().any(|p| p == "zhōng"), "have: {joined}");
    assert!(plain.iter().any(|p| p == "rén"), "have: {joined}");
    assert!(plain.iter().any(|p| p == "guó"), "have: {joined}");
    // '。' is non-Chinese → unk placeholder.
    assert_eq!(plain.last().map(String::as_str), Some("<py_unk>"));
}

#[test]
fn parity_parse_money_docstring_simple() {
    // Python docstring:
    //   "六十四万零一百四十三元一角七分" → 640143.17元
    // Our basic parser doesn't yet handle 角/分 tails; verify the simpler
    // "一百元" case we do support and keep the complex one as a TODO.
    let m = jio::parse_money("一百元").unwrap();
    assert_eq!(m.num, 100.0);
    assert_eq!(m.case, "元");
}

// ────────────────────────────── rule ──────────────────────────────────────

#[test]
fn parity_extract_email_basic() {
    // README example: "please send email to foo@bar.com" → ["foo@bar.com"]
    let r = jio::extract_email("please send email to foo@bar.com");
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].text, "foo@bar.com");
}

#[test]
fn parity_extract_id_card_boundaries() {
    // ID card should match only when surrounded by non-alphanumerics.
    let r = jio::extract_id_card("身份证 11010519900307123X, 谢谢");
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].text, "11010519900307123X");
}

#[test]
fn parity_extract_chinese_runs() {
    // extract_chinese returns contiguous Chinese-char runs.
    assert_eq!(
        jio::extract_chinese("hello 你好 world 测试"),
        vec!["你好".to_string(), "测试".to_string()]
    );
}

#[test]
fn parity_clean_html_entities() {
    // A minimal HTML sample the Python cleanser also handles.
    assert_eq!(jio::clean_html("<p>a &amp; b</p>"), "a & b");
}

#[test]
fn parity_phone_location_beijing_landline() {
    ensure_init();
    let info = jio::phone_location("010-12345678").unwrap();
    assert_eq!(info.phone_type, "landline_phone");
    assert!(info.province.as_deref().unwrap_or("").contains("北京"));
}

#[test]
fn parity_recognize_location_finds_admin_units() {
    ensure_init();
    let r = jio::recognize_location("我出生在广东省广州市海珠区").unwrap();
    let names: Vec<&str> = r.iter().map(|m| m.name.as_str()).collect();
    assert!(names.contains(&"广东省"));
    assert!(names.contains(&"广州市"));
}

// ────────────────────────────── round 7 ───────────────────────────────────

#[test]
fn parity_parse_time_absolute() {
    // Python parse_time docstring:
    //   '2024年3月5日' → ['2024-03-05 00:00:00', '2024-03-05 23:59:59']
    use chrono::{Datelike, Timelike};
    let t = jio::parse_time("2024年3月5日").unwrap();
    assert_eq!(t.start.year(), 2024);
    assert_eq!(t.start.month(), 3);
    assert_eq!(t.start.day(), 5);
    assert_eq!(t.start.hour(), 0);
    assert_eq!(t.end.hour(), 23);
    assert_eq!(t.end.minute(), 59);
}

#[test]
fn parity_parse_time_relative_tomorrow() {
    // Relative days compute against `now`. Verify offset-1 logic with a
    // fixed reference.
    let now = chrono::NaiveDate::from_ymd_opt(2024, 3, 15)
        .unwrap()
        .and_hms_opt(10, 30, 0)
        .unwrap();
    let t = jio::parse_time_with_ref("明天", now).unwrap();
    assert_eq!(
        t.start.date(),
        chrono::NaiveDate::from_ymd_opt(2024, 3, 16).unwrap()
    );
}

#[test]
fn parity_simhash_identical_text() {
    assert_eq!(jio::simhash("今天天气真好"), jio::simhash("今天天气真好"));
}

#[test]
fn parity_simhash_near_duplicate_small_distance() {
    let a = jio::simhash("今天天气真好");
    let b = jio::simhash("今天天气非常好");
    // Same stem "今天天气"+ 好/非常好 — close by SimHash convention.
    assert!(jio::hamming_distance(a, b) < 25);
}

// ────────────────────────────── round 10 ──────────────────────────────────

#[test]
fn parity_parse_time_holiday_guoqing() {
    // Python: '国庆节' with ref 2024-03-15 → 2024-10-01 00:00:00
    use chrono::{Datelike, NaiveDate};
    let now = NaiveDate::from_ymd_opt(2024, 3, 15)
        .unwrap()
        .and_hms_opt(10, 30, 0)
        .unwrap();
    let t = jio::parse_time_with_ref("国庆节", now).unwrap();
    assert_eq!(t.start.year(), 2024);
    assert_eq!(t.start.month(), 10);
    assert_eq!(t.start.day(), 1);
}

#[test]
fn parity_parse_time_range() {
    // Python: '2024年3月5日到8日' → time_span 2024-03-05..2024-03-08
    let t = jio::parse_time("2024年3月5日到8日").unwrap();
    assert_eq!(t.time_type, "time_span");
    assert_eq!(
        t.start.date(),
        chrono::NaiveDate::from_ymd_opt(2024, 3, 5).unwrap()
    );
    assert_eq!(
        t.end.date(),
        chrono::NaiveDate::from_ymd_opt(2024, 3, 8).unwrap()
    );
}

#[test]
fn parity_parse_money_yuan_jiao_fen() {
    // Python docstring: 六十四万零一百四十三元一角七分 → 640143.17元.
    // Our Stage 1 parser doesn't yet consume the Chinese big-number form
    // for the integer part of compound money; verify the simpler
    // "100元5角3分" case we do support.
    let m = jio::parse_money("100元5角3分").unwrap();
    assert!((m.num - 100.53).abs() < 1e-9);
    assert_eq!(m.case, "元");
}

#[test]
fn parity_parse_money_range() {
    let m = jio::parse_money("100-200元").unwrap();
    assert_eq!(m.num, 100.0);
    assert_eq!(m.end_num, Some(200.0));
}

#[test]
fn parity_parse_money_blur() {
    let m = jio::parse_money("约100元").unwrap();
    assert_eq!(m.definition, "blur");
    assert_eq!(m.num, 100.0);
}

#[test]
fn parity_phone_location_cell_mobile() {
    ensure_init();
    // 138 is China Mobile — verified against Python telecom_operator.txt.
    let info = jio::phone_location("13812345678").unwrap();
    assert_eq!(info.phone_type, "cell_phone");
    assert_eq!(info.operator.as_deref(), Some("中国移动"));
}

#[test]
fn parity_sentiment_neutral_for_empty() {
    ensure_init();
    assert!((jio::sentiment_score("").unwrap() - 0.5).abs() < 1e-9);
}

#[test]
fn parity_sentiment_positive_text() {
    ensure_init();
    let s = jio::sentiment_score("今天非常开心,是美好的一天。").unwrap();
    assert!(s > 0.5);
}

#[test]
fn parity_sentiment_negative_text() {
    ensure_init();
    let s = jio::sentiment_score("事故造成严重伤亡,令人悲痛万分。").unwrap();
    assert!(s < 0.5);
}

#[test]
fn parity_pinyin_phrase_override_single_char() {
    ensure_init();
    // 貉: single-char primary reading is "mò" but "一丘之貉" phrase →
    // "hé". Verify the phrase trie wins.
    let r = jio::pinyin("一丘之貉", jio::PinyinFormat::Standard).unwrap();
    let plain: Vec<String> = r
        .iter()
        .map(|e| match e {
            jio::PinyinEntry::Standard(s) => s.clone(),
            _ => String::new(),
        })
        .collect();
    assert_eq!(plain, vec!["yī", "qiū", "zhī", "hé"]);
}

#[test]
fn parity_recognize_location_with_alias() {
    ensure_init();
    // 京 alias → 北京市 (Python: jionlp/rule/rule_pattern.py CHINA_PROVINCE_ALIAS)
    let r = jio::recognize_location("他籍贯京").unwrap();
    assert!(r.iter().any(|m| m.name == "北京市"));
}

#[test]
fn parity_char2num_big_units() {
    assert_eq!(jio::char2num("二十三").unwrap(), 23.0);
    assert_eq!(jio::char2num("一万").unwrap(), 10_000.0);
    assert_eq!(jio::char2num("三千五百万").unwrap(), 35_000_000.0);
}

#[test]
fn parity_num2char_traditional_invoice() {
    // Python docstring example
    assert_eq!(
        jio::num2char("1234", jio::NumStyle::Traditional).unwrap(),
        "壹仟贰佰叁拾肆"
    );
}

#[test]
fn parity_extract_qq() {
    let r = jio::extract_qq("QQ号码 123456789 ,联系");
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].text, "123456789");
}

#[test]
fn parity_extract_ip_multiple() {
    let r = jio::extract_ip_address("server at 192.168.1.1 and 10.0.0.1 online");
    assert_eq!(r.len(), 2);
    let texts: Vec<&str> = r.iter().map(|e| e.text.as_str()).collect();
    assert!(texts.contains(&"192.168.1.1"));
    assert!(texts.contains(&"10.0.0.1"));
}

#[test]
fn parity_id_card_gender_parity() {
    ensure_init();
    // 17th char odd → 男, even → 女.
    let male = jio::parse_id_card("110105199001015671").unwrap().unwrap();
    assert_eq!(male.gender, "男");
    let female = jio::parse_id_card("110105199001012021").unwrap().unwrap();
    assert_eq!(female.gender, "女");
}

#[test]
fn parity_plate_nev_big_pev() {
    // Python docstring: 川A·23047B → PEV, big
    let info = jio::parse_motor_vehicle_licence_plate("川A·23047B").unwrap();
    assert_eq!(info.car_loc, "川A");
    assert_eq!(info.car_type, "PEV");
    assert_eq!(info.car_size, Some("big"));
}

#[test]
fn parity_split_sentence_coarse_quotes() {
    // Python behavior: quotes should attach correctly.
    let text = "他说\"今天天气真好\"。然后离开。";
    let r = jio::split_sentence(text, jio::Criterion::Coarse);
    assert!(!r.is_empty());
    // Last sentence should contain "离开".
    assert!(r.last().unwrap().contains("离开"));
}

#[test]
fn parity_check_all_chinese_char() {
    assert!(jio::check_all_chinese_char("全部中文"));
    assert!(!jio::check_all_chinese_char("中文 with ascii"));
    assert!(!jio::check_all_chinese_char(""));
}

#[test]
fn parity_ts_word_mode_idiom() {
    ensure_init();
    // 速食麵 → 方便面 (Python word-mode tra2sim)
    let r = jio::tra2sim("想喫速食麵", jio::TsMode::Word).unwrap();
    assert!(r.contains("方便面"), "got {r}");
}

#[test]
fn parity_parse_time_timespan() {
    let now = chrono::NaiveDate::from_ymd_opt(2024, 3, 15)
        .unwrap()
        .and_hms_opt(10, 30, 0)
        .unwrap();
    let t = jio::parse_time_with_ref("下午3点到5点", now).unwrap();
    use chrono::Timelike;
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.start.hour(), 15);
    assert_eq!(t.end.hour(), 17);
}

#[test]
fn parity_parse_time_recurring_weekday() {
    let now = chrono::NaiveDate::from_ymd_opt(2024, 3, 15)
        .unwrap()
        .and_hms_opt(10, 30, 0)
        .unwrap();
    let t = jio::parse_time_with_ref("每周一", now).unwrap();
    assert_eq!(
        t.start.date(),
        chrono::NaiveDate::from_ymd_opt(2024, 3, 18).unwrap()
    );
}

// ────────────────────────────── round 12 ──────────────────────────────────
//
// Stage 4 (delta + named period), textaug determinism, phone-location
// non-Beijing cases, pinyin format round-trips, money currency symbols.

#[test]
fn parity_parse_time_delta_days() {
    let now = chrono::NaiveDate::from_ymd_opt(2024, 3, 15)
        .unwrap()
        .and_hms_opt(10, 30, 0)
        .unwrap();
    let t = jio::parse_time_with_ref("三天后", now).unwrap();
    assert_eq!(
        t.start.date(),
        chrono::NaiveDate::from_ymd_opt(2024, 3, 18).unwrap()
    );
}

#[test]
fn parity_parse_time_delta_half_hour() {
    let now = chrono::NaiveDate::from_ymd_opt(2024, 3, 15)
        .unwrap()
        .and_hms_opt(10, 30, 0)
        .unwrap();
    let t = jio::parse_time_with_ref("半小时后", now).unwrap();
    use chrono::Timelike;
    assert_eq!(t.start.hour(), 11);
    assert_eq!(t.start.minute(), 0);
}

#[test]
fn parity_parse_time_delta_weeks_ago() {
    let now = chrono::NaiveDate::from_ymd_opt(2024, 3, 15)
        .unwrap()
        .and_hms_opt(10, 30, 0)
        .unwrap();
    let t = jio::parse_time_with_ref("两周前", now).unwrap();
    assert_eq!(
        t.start.date(),
        chrono::NaiveDate::from_ymd_opt(2024, 3, 1).unwrap()
    );
}

#[test]
fn parity_parse_time_named_this_week() {
    let now = chrono::NaiveDate::from_ymd_opt(2024, 3, 15)
        .unwrap()
        .and_hms_opt(10, 30, 0)
        .unwrap();
    let t = jio::parse_time_with_ref("本周", now).unwrap();
    // Python returns time_point for named weeks.
    assert_eq!(t.time_type, "time_point");
    assert_eq!(
        t.start.date(),
        chrono::NaiveDate::from_ymd_opt(2024, 3, 11).unwrap()
    );
}

#[test]
fn parity_parse_time_named_last_month() {
    let now = chrono::NaiveDate::from_ymd_opt(2024, 3, 15)
        .unwrap()
        .and_hms_opt(10, 30, 0)
        .unwrap();
    let t = jio::parse_time_with_ref("上个月", now).unwrap();
    assert_eq!(t.start.date().month(), 2);
    use chrono::Datelike;
    assert_eq!(t.end.date().day(), 29); // 2024 leap year
}

#[test]
fn parity_parse_time_named_next_quarter() {
    let now = chrono::NaiveDate::from_ymd_opt(2024, 3, 15)
        .unwrap()
        .and_hms_opt(10, 30, 0)
        .unwrap();
    let t = jio::parse_time_with_ref("下季度", now).unwrap();
    assert_eq!(
        t.start.date(),
        chrono::NaiveDate::from_ymd_opt(2024, 4, 1).unwrap()
    );
    assert_eq!(
        t.end.date(),
        chrono::NaiveDate::from_ymd_opt(2024, 6, 30).unwrap()
    );
}

#[test]
fn parity_parse_time_named_last_year() {
    let now = chrono::NaiveDate::from_ymd_opt(2024, 3, 15)
        .unwrap()
        .and_hms_opt(10, 30, 0)
        .unwrap();
    let t = jio::parse_time_with_ref("去年", now).unwrap();
    use chrono::Datelike;
    assert_eq!(t.start.date().year(), 2023);
    assert_eq!(t.end.date().year(), 2023);
}

#[test]
fn parity_parse_money_currency_symbols() {
    let y = jio::parse_money("￥100").unwrap();
    assert_eq!(y.case, "元");
    let d = jio::parse_money("$100.5").unwrap();
    assert_eq!(d.case, "美元");
    let e = jio::parse_money("€50").unwrap();
    assert_eq!(e.case, "欧元");
}

#[test]
fn parity_phone_location_unicom() {
    ensure_init();
    // 130 is 中国联通 per `telecom_operator.txt`.
    let info = jio::phone_location("13012345678").unwrap();
    assert_eq!(info.operator.as_deref(), Some("中国联通"));
}

#[test]
fn parity_phone_location_telecom() {
    ensure_init();
    // 133 is 中国电信.
    let info = jio::phone_location("13312345678").unwrap();
    assert_eq!(info.operator.as_deref(), Some("中国电信"));
}

#[test]
fn parity_pinyin_simple_format_roundtrip() {
    ensure_init();
    let r = jio::pinyin("中国", jio::PinyinFormat::Simple).unwrap();
    let s: Vec<String> = r
        .iter()
        .map(|e| match e {
            jio::PinyinEntry::Simple(s) => s.clone(),
            _ => String::new(),
        })
        .collect();
    assert_eq!(s, vec!["zhong1", "guo2"]);
}

#[test]
fn parity_pinyin_detail_format_components() {
    ensure_init();
    let r = jio::pinyin("中", jio::PinyinFormat::Detail).unwrap();
    match &r[0] {
        jio::PinyinEntry::Detail(d) => {
            assert_eq!(d.consonant, "zh");
            assert_eq!(d.vowel, "ong");
            assert_eq!(d.tone, "1");
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn parity_swap_char_determinism() {
    // Same seed → identical outputs. This pins the PRNG discipline.
    let a = jio::swap_char_position("中华人民共和国是美好的家园", 3, 0.3, 42, 1.0);
    let b = jio::swap_char_position("中华人民共和国是美好的家园", 3, 0.3, 42, 1.0);
    assert_eq!(a, b);
    assert_eq!(a.len(), 3);
}

#[test]
fn parity_homophone_substitution_preserves_length() {
    ensure_init();
    let src = "今天天气真好";
    let r = jio::homophone_substitution(src, 3, 0.5, 42).unwrap();
    for v in r {
        assert_eq!(v.chars().count(), src.chars().count());
    }
}

#[test]
fn parity_random_add_delete_nonempty() {
    let r = jio::random_add_delete("今天天气真好我要去公园散步", 3, 0.3, 0.1, 42);
    assert!(!r.is_empty());
    for v in &r {
        assert!(!v.is_empty());
    }
}

#[test]
fn parity_extract_summary_preserves_order() {
    ensure_init();
    let text = "第一句话。第二句话。第三句话。第四句话。";
    let out = jio::extract_summary(text, 3).unwrap();
    for w in out.windows(2) {
        assert!(w[0].position < w[1].position);
    }
}

#[test]
fn parity_sentiment_adverb_strengthens_positive() {
    ensure_init();
    let a = jio::sentiment_score("今天开心").unwrap();
    let b = jio::sentiment_score("今天非常开心").unwrap();
    assert!(b >= a);
}

#[test]
fn parity_check_all_arabic_num() {
    assert!(jio::check_all_arabic_num("12345"));
    assert!(!jio::check_all_arabic_num("12a45"));
    assert!(!jio::check_all_arabic_num(""));
}

#[test]
fn parity_extract_url_with_trailing_punct() {
    let r = jio::extract_url("访问 https://example.com/path?q=1。然后...");
    assert_eq!(r.len(), 1);
    assert!(r[0].text.starts_with("https://"));
    // Trailing "。" should not be captured.
    assert!(!r[0].text.contains("。"));
}

#[test]
fn parity_extract_parentheses_chinese_and_ascii() {
    // Mixed ASCII and Chinese brackets, with nesting.
    let r = jio::extract_parentheses("a (b) c【标题】d", "()[]【】");
    let texts: Vec<&str> = r.iter().map(|e| e.text.as_str()).collect();
    assert!(texts.contains(&"(b)"));
    assert!(texts.contains(&"【标题】"));
}

#[test]
fn parity_clean_html_removes_script_tags() {
    let r = jio::clean_html("<html><script>alert(1)</script>hi</html>");
    assert!(!r.contains("alert"));
    assert!(r.contains("hi"));
}

// ────────────────────────────── round 13 ──────────────────────────────────
//
// bpe byte-level codec, Stage-5 fuzzy time, summary MMR, and additional
// boundary cases from Python docstrings / test corpus.

#[test]
fn parity_bpe_roundtrip_scripts() {
    // Byte-level BPE must be lossless across scripts.
    for src in [
        "Hello 世界 🌍 テスト",
        "メトロ",
        "今天天气真好",
        "Mixed! punctuation: 。，",
    ] {
        assert_eq!(jio::bpe_decode(&jio::bpe_encode(src)), src, "src={src}");
    }
}

#[test]
fn parity_bpe_control_byte_remap() {
    // Tab (0x09) must remap to a code point >= U+0100.
    let enc = jio::bpe_encode("\t");
    let c = enc.chars().next().unwrap() as u32;
    assert!(c >= 0x0100);
    assert_eq!(jio::bpe_decode(&enc), "\t");
}

#[test]
fn parity_parse_time_fuzzy_gangcai() {
    let now = chrono::NaiveDate::from_ymd_opt(2024, 3, 15)
        .unwrap()
        .and_hms_opt(10, 30, 0)
        .unwrap();
    let t = jio::parse_time_with_ref("刚才", now).unwrap();
    assert_eq!(t.definition, "blur");
    assert!(t.end <= now);
}

#[test]
fn parity_parse_time_fuzzy_mashangjiu() {
    let now = chrono::NaiveDate::from_ymd_opt(2024, 3, 15)
        .unwrap()
        .and_hms_opt(10, 30, 0)
        .unwrap();
    let t = jio::parse_time_with_ref("马上", now).unwrap();
    assert_eq!(t.definition, "blur");
    assert!(t.end > now);
}

#[test]
fn parity_parse_time_fuzzy_longest_match() {
    let now = chrono::NaiveDate::from_ymd_opt(2024, 3, 15)
        .unwrap()
        .and_hms_opt(10, 30, 0)
        .unwrap();
    // "不久之前" (4 chars) should win over "不久" or "不久前" (2-3 chars).
    let t = jio::parse_time_with_ref("不久之前", now).unwrap();
    assert_eq!(t.definition, "blur");
}

#[test]
fn parity_extract_summary_mmr_lambda_boundaries() {
    ensure_init();
    let text = "第一句话。第二句话。第三句话。";
    // λ=1 matches basic top-k.
    let basic = jio::extract_summary(text, 2).unwrap();
    let mmr = jio::extract_summary_mmr(text, 2, 1.0).unwrap();
    let bp: Vec<_> = basic.iter().map(|s| s.position).collect();
    let mp: Vec<_> = mmr.iter().map(|s| s.position).collect();
    assert_eq!(bp, mp);

    // λ=0 must not panic and returns top_k.
    let diverse = jio::extract_summary_mmr(text, 2, 0.0).unwrap();
    assert_eq!(diverse.len(), 2);
}

#[test]
fn parity_parse_time_abs_slash_format() {
    // Python parses "2024/3/5" equivalently to "2024年3月5日".
    let t = jio::parse_time("2024/3/5").unwrap();
    use chrono::Datelike;
    assert_eq!(t.start.year(), 2024);
    assert_eq!(t.start.month(), 3);
    assert_eq!(t.start.day(), 5);
}

#[test]
fn parity_parse_time_two_digit_year_future() {
    // Python: "24年3月5日" → 2024.
    let t = jio::parse_time("24年3月5日").unwrap();
    use chrono::Datelike;
    assert_eq!(t.start.year(), 2024);
}

#[test]
fn parity_parse_time_two_digit_year_past() {
    // Python: "98年3月5日" → 1998.
    let t = jio::parse_time("98年3月5日").unwrap();
    use chrono::Datelike;
    assert_eq!(t.start.year(), 1998);
}

#[test]
fn parity_char_radical_ascii_fallback() {
    // ASCII chars fall back to <cr_unk> placeholder.
    ensure_init();
    let r = jio::char_radical("A").unwrap();
    assert_eq!(r[0].radical, "<cr_unk>");
}

#[test]
fn parity_split_sentence_fine_multipunct() {
    // "……" should stay together, not split into two "…" (fine mode).
    let r = jio::split_sentence("这是第一句……这是第二句。", jio::Criterion::Fine);
    assert_eq!(r, vec!["这是第一句……", "这是第二句。"]);
}

#[test]
fn parity_remove_stopwords_preserves_plain_words() {
    ensure_init();
    // Non-stopwords should pass through untouched.
    let input: Vec<String> = ["机器学习", "深度学习", "神经网络"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let r = jio::remove_stopwords(&input, jio::RemoveOpts::default()).unwrap();
    assert_eq!(r, input);
}

#[test]
fn parity_check_cell_phone_prefix_range() {
    // Python CELL_PHONE_CHECK_PATTERN requires 1[3-9]x.
    assert!(jio::check_cell_phone("13912345678"));
    assert!(!jio::check_cell_phone("12312345678"));
    assert!(!jio::check_cell_phone("10012345678"));
}

#[test]
fn parity_parse_money_comma_and_decimal() {
    let m = jio::parse_money("1,234.56元").unwrap();
    assert!((m.num - 1234.56).abs() < 1e-9);
    assert_eq!(m.case, "元");
}

#[test]
fn parity_parse_money_bare_default_yuan() {
    // Python default_unit='元' when no currency is given.
    let m = jio::parse_money_with_default("1000", "元").unwrap();
    assert_eq!(m.num, 1000.0);
    assert_eq!(m.case, "元");
}

#[test]
fn parity_extract_chinese_filters_ascii_and_digits() {
    assert_eq!(
        jio::extract_chinese("hello 中文 123 world 测试 !"),
        vec!["中文".to_string(), "测试".to_string()]
    );
}

#[test]
fn parity_check_any_chinese_char_edge() {
    // Empty → false (the Python behavior).
    assert!(!jio::check_any_chinese_char(""));
}

#[test]
fn parity_ts_sim2tra_idiom_roundtrip() {
    ensure_init();
    // Word-mode round-trip is lossy in Python too (multi-reading chars),
    // but character-mode should be stable on pure simplified text.
    let src = "今天天气好晴朗";
    let to_tra = jio::sim2tra(src, jio::TsMode::Char).unwrap();
    let back = jio::tra2sim(&to_tra, jio::TsMode::Char).unwrap();
    assert_eq!(back, src);
}

#[test]
fn parity_simhash_similarity_self_is_one() {
    let h = jio::simhash("some text");
    assert!((jio::simhash_similarity(h, h) - 1.0).abs() < 1e-9);
}

// ────────────────────────────── round 14 ──────────────────────────────────
//
// Stage 6 lunar holidays, TextRank keyphrase, and final edge cases to
// bring the parity suite past 100.

#[test]
fn parity_lunar_spring_festival_2024() {
    let t = jio::parse_time("2024年春节").unwrap();
    assert_eq!(
        t.start.date(),
        chrono::NaiveDate::from_ymd_opt(2024, 2, 10).unwrap()
    );
}

#[test]
fn parity_lunar_mid_autumn_alias() {
    // "中秋" (without 节) should still resolve.
    let now = chrono::NaiveDate::from_ymd_opt(2024, 3, 15)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let t = jio::parse_time_with_ref("中秋", now).unwrap();
    assert_eq!(
        t.start.date(),
        chrono::NaiveDate::from_ymd_opt(2024, 9, 17).unwrap()
    );
}

#[test]
fn parity_lunar_new_year_eve() {
    // Python treats `2025年除夕` as the lunar new year's eve of lunar
    // year 2025 (solar 2026-02-16).
    let t = jio::parse_time("2025年除夕").unwrap();
    assert_eq!(
        t.start.date(),
        chrono::NaiveDate::from_ymd_opt(2026, 2, 16).unwrap()
    );
}

#[test]
fn parity_lunar_dragon_boat_2023() {
    let t = jio::parse_time("2023年端午节").unwrap();
    assert_eq!(
        t.start.date(),
        chrono::NaiveDate::from_ymd_opt(2023, 6, 22).unwrap()
    );
}

#[test]
fn parity_lunar_chongyang_with_clock() {
    let now = chrono::NaiveDate::from_ymd_opt(2024, 3, 15)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let t = jio::parse_time_with_ref("重阳节上午10点", now).unwrap();
    use chrono::Timelike;
    assert_eq!(t.start.hour(), 10);
}

#[test]
fn parity_lunar_out_of_range() {
    // Round 15: we now support 1900-2100. Only years outside that span
    // return None.
    assert!(jio::parse_time("1899年春节").is_none());
    assert!(jio::parse_time("2101年春节").is_none());
}

#[test]
fn parity_lunar_full_coverage_1900_2100() {
    // Full-range smoke: 2019, 2050, 1950 all resolve.
    use chrono::Datelike;
    for (y, expect_m) in [(2019, 2u32), (2050, 1u32), (1950, 2u32)] {
        let t = jio::parse_time(&format!("{y}年春节")).unwrap();
        assert_eq!(t.start.year(), y);
        assert_eq!(t.start.month(), expect_m);
    }
}

#[test]
fn parity_chinese_numeral_year_lingsan() {
    // 零三 → 2003 per 2-digit expansion rule.
    use chrono::Datelike;
    let t = jio::parse_time("零三年元宵节").unwrap();
    assert_eq!(t.start.year(), 2003);
    assert_eq!(t.start.month(), 2);
    assert_eq!(t.start.day(), 15);
}

#[test]
fn parity_clock_banxiao() {
    // "8点半" = 08:30
    let now = chrono::NaiveDate::from_ymd_opt(2024, 3, 15)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    use chrono::Timelike;
    let t = jio::parse_time_with_ref("8点半", now).unwrap();
    assert_eq!(t.start.hour(), 8);
    assert_eq!(t.start.minute(), 30);
}

#[test]
fn parity_user_regression_lingsan_yuanxiao() {
    // User-reported example: 零三年元宵节晚上8点半 → 2003-02-15 20:30:00
    use chrono::Timelike;
    let t = jio::parse_time("零三年元宵节晚上8点半").unwrap();
    assert_eq!(
        t.start.date(),
        chrono::NaiveDate::from_ymd_opt(2003, 2, 15).unwrap()
    );
    assert_eq!(t.start.hour(), 20);
    assert_eq!(t.start.minute(), 30);
}

#[test]
fn parity_textrank_nonempty() {
    ensure_init();
    let text = "机器学习是人工智能的一个分支。机器学习研究数据。机器学习应用广泛。";
    let r = jio::extract_keyphrase_textrank(text, 3, 2, 3).unwrap();
    assert!(!r.is_empty());
}

#[test]
fn parity_textrank_sorted_descending() {
    ensure_init();
    let r = jio::extract_keyphrase_textrank("北京是中国首都。北京有名胜古迹。", 5, 2, 3).unwrap();
    for w in r.windows(2) {
        assert!(w[0].weight >= w[1].weight);
    }
}

#[test]
fn parity_textrank_empty() {
    ensure_init();
    assert!(jio::extract_keyphrase_textrank("", 5, 2, 3)
        .unwrap()
        .is_empty());
}

#[test]
fn parity_parse_time_fuzzy_entire_table() {
    let now = chrono::NaiveDate::from_ymd_opt(2024, 3, 15)
        .unwrap()
        .and_hms_opt(10, 30, 0)
        .unwrap();
    // Every fuzzy keyword the parser recognizes should return Some(blur).
    for kw in [
        "刚才",
        "刚刚",
        "不久前",
        "最近",
        "近期",
        "晚些时候",
        "马上",
        "立刻",
        "稍后",
        "即将",
        "过一会儿",
        "一会儿",
        "等会儿",
    ] {
        let t = jio::parse_time_with_ref(kw, now)
            .unwrap_or_else(|| panic!("parser returned None for '{kw}'"));
        assert_eq!(t.definition, "blur", "kw={kw}");
    }
}

#[test]
fn parity_sentiment_neutral_edge_cases() {
    ensure_init();
    // Text with no sentiment vocabulary → 0.5 (neutral).
    let s = jio::sentiment_score("今天是星期三").unwrap();
    assert!((s - 0.5).abs() < 0.05, "expected ~0.5, got {s}");
}

#[test]
fn parity_extract_id_card_filters_18_chars() {
    // Must be exactly 18 chars, not 17 or 19.
    assert!(jio::extract_id_card("张三: 11010519900307123 未完成").is_empty());
}

#[test]
fn parity_extract_email_domain_dot_required() {
    // No TLD → not an email.
    assert!(jio::extract_email("foo@bar 缺少顶级域名").is_empty());
}

#[test]
fn parity_num2char_single_digit_zero() {
    assert_eq!(jio::num2char("0", jio::NumStyle::Simplified).unwrap(), "零");
}

#[test]
fn parity_char2num_single_char() {
    assert_eq!(jio::char2num("五").unwrap(), 5.0);
}

#[test]
fn parity_extract_parentheses_unmatched_is_skipped() {
    // Mismatched brackets are lenient: no crash, just no entry.
    let r = jio::extract_parentheses("没有闭合 (开始", "()");
    assert!(r.is_empty());
}

#[test]
fn parity_clean_html_numeric_entity() {
    // &#65; → "A", &#x4E2D; → "中"
    assert_eq!(jio::clean_html("&#65;"), "A");
    assert_eq!(jio::clean_html("&#x4E2D;"), "中");
}

#[test]
fn parity_check_all_chinese_char_rejects_mixed() {
    assert!(!jio::check_all_chinese_char("中文123"));
    assert!(!jio::check_all_chinese_char("123"));
}

#[test]
fn parity_lexicon_ner_leftmost_longest() {
    // LexiconNer with 北京 + 北京大学 → leftmost-longest picks 北京大学.
    let mut lex: rustc_hash::FxHashMap<String, Vec<String>> = rustc_hash::FxHashMap::default();
    lex.insert(
        "Univ".to_string(),
        vec!["北京".to_string(), "北京大学".to_string()],
    );
    let ner = jio::LexiconNer::from_map(&lex).unwrap();
    let r = ner.recognize("就读于北京大学");
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].text, "北京大学");
}

// ───────────────────────── parse_time — round 16 cases ────────────────────

fn r16_ref() -> chrono::NaiveDateTime {
    chrono::NaiveDate::from_ymd_opt(2021, 6, 15)
        .unwrap()
        .and_hms_opt(10, 0, 0)
        .unwrap()
}

#[test]
fn parity_parse_time_lunar_chinese_year_clock() {
    // Python: parse_time('零三年元宵节晚上8点半') → 2003-02-15 20:30:00.
    let t = jio::parse_time_with_ref("零三年元宵节晚上8点半", r16_ref()).unwrap();
    assert_eq!(t.time_type, "time_point");
    assert_eq!(t.start.to_string(), "2003-02-15 20:30:00");
}

#[test]
fn parity_parse_time_year_span_quarters() {
    // Python: parse_time('2021年前两个季度') → time_span 2021 Q1-Q2.
    let t = jio::parse_time_with_ref("2021年前两个季度", r16_ref()).unwrap();
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.start.to_string(), "2021-01-01 00:00:00");
    assert_eq!(t.end.to_string(), "2021-06-30 23:59:59");
}

#[test]
fn parity_parse_time_cong_chinese_day_range() {
    // Python: parse_time('从2018年12月九号到十五号') → 2018-12-09..2018-12-15.
    let t = jio::parse_time_with_ref("从2018年12月九号到十五号", r16_ref()).unwrap();
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.start.to_string(), "2018-12-09 00:00:00");
    assert_eq!(t.end.to_string(), "2018-12-15 23:59:59");
}

#[test]
fn parity_parse_time_nth_weekday_festival() {
    // Python: parse_time('2019年感恩节') → 2019-11-28 (4th Thursday of Nov).
    let t = jio::parse_time_with_ref("2019年感恩节", r16_ref()).unwrap();
    assert_eq!(t.time_type, "time_point");
    assert_eq!(t.start.to_string(), "2019-11-28 00:00:00");
}

#[test]
fn parity_parse_time_recurring_weekly_with_clock_range() {
    // Python: parse_time('每周六上午9点到11点') →
    //   time_period, delta={day:7}, point covers 09:00..11:00.
    let t = jio::parse_time_with_ref("每周六上午9点到11点", r16_ref()).unwrap();
    assert_eq!(t.time_type, "time_period");
    let d = t.delta.as_ref().expect("delta populated for time_period");
    match d.day {
        Some(jio::DeltaValue::Single(n)) => assert_eq!(n, 7.0),
        _ => panic!("expected weekly cadence {{day: 7}}, got {:?}", d.day),
    }
    let p = t.period.as_ref().expect("period populated");
    assert_eq!(p.point_time.len(), 1);
    assert_eq!(p.point_time[0].0.format("%H:%M:%S").to_string(), "09:00:00");
    assert_eq!(p.point_time[0].1.format("%H:%M:%S").to_string(), "11:00:00");
}

#[test]
fn parity_parse_time_delta_day_range() {
    // Python: parse_time('30~90日') →
    //   time_delta, {day: [30, 90]}.
    let t = jio::parse_time_with_ref("30~90日", r16_ref()).unwrap();
    assert_eq!(t.time_type, "time_delta");
    let d = t.delta.as_ref().expect("delta populated");
    match d.day {
        Some(jio::DeltaValue::Range(lo, hi)) => {
            assert_eq!(lo, 30.0);
            assert_eq!(hi, 90.0);
        }
        _ => panic!("expected day Range(30, 90), got {:?}", d.day),
    }
}

// ───────────────────────── parse_time — round 17 cases ────────────────────

fn r17_ref() -> chrono::NaiveDateTime {
    // Friday 2024-03-15.
    chrono::NaiveDate::from_ymd_opt(2024, 3, 15)
        .unwrap()
        .and_hms_opt(10, 30, 0)
        .unwrap()
}

#[test]
fn parity_parse_time_eight_digit_ymd() {
    let t = jio::parse_time_with_ref("20210901", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_point");
    assert_eq!(t.start.to_string(), "2021-09-01 00:00:00");
    assert_eq!(t.end.to_string(), "2021-09-01 23:59:59");
}

#[test]
fn parity_parse_time_year_solar_season() {
    let t = jio::parse_time_with_ref("2021年第1季度", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.start.to_string(), "2021-01-01 00:00:00");
    assert_eq!(t.end.to_string(), "2021-03-31 23:59:59");
}

#[test]
fn parity_parse_time_year_blur_boundary() {
    let t = jio::parse_time_with_ref("2024年末", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.definition, "blur");
    assert_eq!(t.start.to_string(), "2024-11-01 00:00:00");
    assert_eq!(t.end.to_string(), "2024-12-31 23:59:59");
}

#[test]
fn parity_parse_time_limit_month_day() {
    let t = jio::parse_time_with_ref("下个月9号", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_point");
    // 2024-03 + 1 month = 2024-04.
    assert_eq!(t.start.to_string(), "2024-04-09 00:00:00");
}

#[test]
fn parity_parse_time_century_full() {
    // Python treats 20世纪 = 1900..1999 (convention starts at year 1900,
    // not 1901).
    let t = jio::parse_time_with_ref("20世纪", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.start.to_string(), "1900-01-01 00:00:00");
    assert_eq!(t.end.to_string(), "1999-12-31 23:59:59");
}

#[test]
fn parity_parse_time_century_decade() {
    let t = jio::parse_time_with_ref("20世纪二十年代", r17_ref()).unwrap();
    assert_eq!(t.start.to_string(), "1920-01-01 00:00:00");
    assert_eq!(t.end.to_string(), "1929-12-31 23:59:59");
}

#[test]
fn parity_parse_time_standalone_weekday() {
    // now = Friday 2024-03-15; 周一 → next Monday 2024-03-18.
    let t = jio::parse_time_with_ref("周一", r17_ref()).unwrap();
    assert_eq!(t.start.to_string(), "2024-03-18 00:00:00");
}

#[test]
fn parity_parse_time_named_weekday() {
    // 下周五 = 2024-03-22.
    let t = jio::parse_time_with_ref("下周五", r17_ref()).unwrap();
    assert_eq!(t.start.to_string(), "2024-03-22 00:00:00");
}

#[test]
fn parity_parse_time_year_week() {
    // 2024 week 5: Monday = 2024-01-29.
    let t = jio::parse_time_with_ref("2024年第5周", r17_ref()).unwrap();
    assert_eq!(t.start.to_string(), "2024-01-29 00:00:00");
    assert_eq!(t.end.to_string(), "2024-02-04 23:59:59");
}

#[test]
fn parity_parse_time_nth_weekday_in_month() {
    // 2024-10 first Friday = 2024-10-04.
    let t = jio::parse_time_with_ref("2024年10月的第一个周五", r17_ref()).unwrap();
    assert_eq!(t.start.to_string(), "2024-10-04 00:00:00");
}

#[test]
fn parity_parse_time_year_day_ordinal() {
    let t = jio::parse_time_with_ref("2024年第100天", r17_ref()).unwrap();
    // 2024-01-01 + 99 days = 2024-04-09.
    assert_eq!(t.start.to_string(), "2024-04-09 00:00:00");
}

#[test]
fn parity_parse_time_special_now() {
    let t = jio::parse_time_with_ref("现在", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_point");
    assert_eq!(t.start.to_string(), "2024-03-15 10:30:00");
    assert_eq!(t.end, t.start);
}

#[test]
fn parity_parse_time_special_all_day() {
    let t = jio::parse_time_with_ref("全天", r17_ref()).unwrap();
    assert_eq!(t.start.to_string(), "2024-03-15 00:00:00");
    assert_eq!(t.end.to_string(), "2024-03-15 23:59:59");
}

#[test]
fn parity_parse_time_jinming_two_days() {
    let t = jio::parse_time_with_ref("今明两天", r17_ref()).unwrap();
    assert_eq!(t.start.to_string(), "2024-03-15 00:00:00");
    assert_eq!(t.end.to_string(), "2024-03-16 23:59:59");
}

#[test]
fn parity_parse_time_super_blur_past_two_days() {
    // Python `前两天` = [now-7d, now-2d] whole days.
    let t = jio::parse_time_with_ref("前两天", r17_ref()).unwrap();
    assert_eq!(t.definition, "blur");
    assert_eq!(t.start.to_string(), "2024-03-08 00:00:00");
    assert_eq!(t.end.to_string(), "2024-03-13 23:59:59");
}

#[test]
fn parity_parse_time_super_blur_future_three_days() {
    // Python `未来三天` = [now, now+3d] preserving now's time.
    let t = jio::parse_time_with_ref("未来三天", r17_ref()).unwrap();
    assert_eq!(t.start.to_string(), "2024-03-15 10:30:00");
    assert_eq!(t.end.to_string(), "2024-03-18 10:30:00");
}

// ───────────────────────── parse_time — round 18 cases ────────────────────

#[test]
fn parity_parse_time_lunar_mid_year() {
    // Python: parse_time('2012年农历正月十九') → 2012-02-10.
    let t = jio::parse_time_with_ref("2012年农历正月十九", r17_ref()).unwrap();
    assert_eq!(t.start.to_string(), "2012-02-10 00:00:00");
}

#[test]
fn parity_parse_time_lunar_standalone() {
    let t = jio::parse_time_with_ref("腊月二十八", r17_ref()).unwrap();
    // 2024 now → 2024 lunar 12/28 maps to 2025-01-27.
    assert_eq!(t.start.to_string(), "2025-01-27 00:00:00");
}

#[test]
fn parity_parse_time_solar_term_with_year() {
    let t = jio::parse_time_with_ref("2021年清明", r17_ref()).unwrap();
    assert_eq!(t.start.to_string(), "2021-04-05 00:00:00");
}

#[test]
fn parity_parse_time_solar_term_standalone() {
    // now.year() = 2024; 立春 2024 = Feb 4.
    let t = jio::parse_time_with_ref("立春", r17_ref()).unwrap();
    assert_eq!(t.start.to_string(), "2024-02-04 00:00:00");
}

#[test]
fn parity_parse_time_season_with_year() {
    // Python: `2021年春天` uses solar-term boundaries (立春..立夏-1).
    let t = jio::parse_time_with_ref("2021年春天", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.start.to_string(), "2021-02-03 00:00:00");
    assert_eq!(t.end.to_string(), "2021-05-04 23:59:59");
}

#[test]
fn parity_parse_time_season_limit_year() {
    // Python solar-term boundaries: 立夏 2023 = May 6.
    let t = jio::parse_time_with_ref("去年夏季", r17_ref()).unwrap();
    assert_eq!(t.start.to_string(), "2023-05-06 00:00:00");
    assert_eq!(t.end.to_string(), "2023-08-07 23:59:59");
}

#[test]
fn parity_parse_time_limit_year_festival_fixed() {
    // 今年儿童节 with now=2024 → 2024-06-01.
    let t = jio::parse_time_with_ref("今年儿童节", r17_ref()).unwrap();
    assert_eq!(t.start.to_string(), "2024-06-01 00:00:00");
}

#[test]
fn parity_parse_time_limit_year_festival_lunar() {
    // 明年端午 with now=2024 → lunar 2025/5/5 = Gregorian 2025-05-31.
    let t = jio::parse_time_with_ref("明年端午", r17_ref()).unwrap();
    assert_eq!(t.start.to_string(), "2025-05-31 00:00:00");
}

#[test]
fn parity_parse_time_limit_year_festival_nth_weekday() {
    // 今年母亲节 = 2nd Sunday of May 2024 = 2024-05-12.
    let t = jio::parse_time_with_ref("今年母亲节", r17_ref()).unwrap();
    assert_eq!(t.start.to_string(), "2024-05-12 00:00:00");
}

// ───────────────────────── parse_time — round 19 cases ────────────────────

#[test]
fn parity_parse_time_pure_delta_year() {
    let t = jio::parse_time_with_ref("3年", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_delta");
    match t.delta.as_ref().unwrap().year {
        Some(jio::DeltaValue::Single(n)) => assert_eq!(n, 3.0),
        _ => panic!("expected year Single(3)"),
    }
}

#[test]
fn parity_parse_time_pure_delta_month_chinese() {
    let t = jio::parse_time_with_ref("两个月", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_delta");
    match t.delta.as_ref().unwrap().month {
        Some(jio::DeltaValue::Single(n)) => assert_eq!(n, 2.0),
        _ => panic!("expected month Single(2)"),
    }
}

#[test]
fn parity_parse_time_pure_delta_blur_tens_of_days() {
    // Python `几十` = [20, 100].
    let t = jio::parse_time_with_ref("几十天", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_delta");
    assert_eq!(t.definition, "blur");
    match t.delta.as_ref().unwrap().day {
        Some(jio::DeltaValue::Range(lo, hi)) => {
            assert_eq!(lo, 20.0);
            assert_eq!(hi, 100.0);
        }
        _ => panic!("expected day Range(20, 100)"),
    }
}

#[test]
fn parity_parse_time_delta_to_span_future() {
    // Python `再过五天` → open-ended [now+5d, inf].
    let t = jio::parse_time_with_ref("再过五天", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.start.to_string(), "2024-03-20 10:30:00");
}

#[test]
fn parity_parse_time_delta_inner_span() {
    // Python `5分钟内` → span [now, now+5min].
    let t = jio::parse_time_with_ref("5分钟内", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.start.to_string(), "2024-03-15 10:30:00");
    assert_eq!(t.end.to_string(), "2024-03-15 10:35:00");
}

#[test]
fn parity_parse_time_delta_open_ended_subday() {
    // Round 32: `之前` now emits `time_span` (Python parity). The span
    // runs from 48h ago to now.
    let t = jio::parse_time_with_ref("48小时之前", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.start.to_string(), "2024-03-13 10:30:00");
    assert_eq!(t.end.to_string(), "2024-03-15 10:30:00");
}

#[test]
fn parity_parse_time_workday_delta() {
    // now = Fri 2024-03-15; 1 workday before = Thu 2024-03-14.
    let t = jio::parse_time_with_ref("1个工作日前", r17_ref()).unwrap();
    assert_eq!(t.start.to_string(), "2024-03-14 00:00:00");
}

#[test]
fn parity_parse_time_pure_delta_workday() {
    let t = jio::parse_time_with_ref("3个工作日", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_delta");
    match t.delta.as_ref().unwrap().workday {
        Some(jio::DeltaValue::Single(n)) => assert_eq!(n, 3.0),
        _ => panic!("expected workday Single(3)"),
    }
}

#[test]
fn parity_parse_time_pure_delta_range_cn() {
    // 两三天 → {day: [2, 3]}.
    let t = jio::parse_time_with_ref("两三天", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_delta");
    assert_eq!(t.definition, "blur");
    match t.delta.as_ref().unwrap().day {
        Some(jio::DeltaValue::Range(lo, hi)) => {
            assert_eq!(lo, 2.0);
            assert_eq!(hi, 3.0);
        }
        _ => panic!("expected day Range(2, 3)"),
    }
}

// ───────────────────────── parse_time — round 20 cases ────────────────────

#[test]
fn parity_parse_time_blur_hour_morning() {
    // Python classifies blur-hour phrases as time_point (a named part of
    // the day) with blur definition.
    let t = jio::parse_time_with_ref("早上", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_point");
    assert_eq!(t.start.to_string(), "2024-03-15 06:00:00");
    assert_eq!(t.end.to_string(), "2024-03-15 09:59:59");
}

#[test]
fn parity_parse_time_blur_hour_with_day_prefix() {
    let t = jio::parse_time_with_ref("明天晚上", r17_ref()).unwrap();
    assert_eq!(t.start.to_string(), "2024-03-16 18:00:00");
    assert_eq!(t.end.to_string(), "2024-03-16 23:59:59");
}

#[test]
fn parity_parse_time_approx_clock() {
    let t = jio::parse_time_with_ref("约9点", r17_ref()).unwrap();
    assert_eq!(t.definition, "blur");
    assert_eq!(t.start.format("%H:%M").to_string(), "09:00");
}

#[test]
fn parity_parse_time_super_blur_hms() {
    // Python super_blur_two_hms: `前两个小时` → [now-6h, now-2h] hour-precision.
    let t = jio::parse_time_with_ref("前两个小时", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.start.to_string(), "2024-03-15 04:00:00");
    assert_eq!(t.end.to_string(), "2024-03-15 08:59:59");
}

#[test]
fn parity_parse_time_recurring_hourly() {
    let t = jio::parse_time_with_ref("每小时", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_period");
    match t.delta.as_ref().unwrap().hour {
        Some(jio::DeltaValue::Single(n)) => assert_eq!(n, 1.0),
        _ => panic!("expected hour Single(1)"),
    }
}

#[test]
fn parity_parse_time_recurring_every_n_min() {
    let t = jio::parse_time_with_ref("每30分钟", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_period");
    match t.delta.as_ref().unwrap().minute {
        Some(jio::DeltaValue::Single(n)) => assert_eq!(n, 30.0),
        _ => panic!("expected minute Single(30)"),
    }
}

#[test]
fn parity_parse_time_recurring_gap() {
    // 每隔一天 = cadence 2 days.
    let t = jio::parse_time_with_ref("每隔一天", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_period");
    match t.delta.as_ref().unwrap().day {
        Some(jio::DeltaValue::Single(n)) => assert_eq!(n, 2.0),
        _ => panic!("expected day Single(2)"),
    }
}

#[test]
fn parity_parse_time_recurring_yearly_festival() {
    let t = jio::parse_time_with_ref("每年春节", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_period");
    // 2024 春节 = 2024-02-10.
    assert_eq!(t.start.to_string(), "2024-02-10 00:00:00");
    match t.delta.as_ref().unwrap().year {
        Some(jio::DeltaValue::Single(n)) => assert_eq!(n, 1.0),
        _ => panic!("expected year Single(1)"),
    }
}

#[test]
fn parity_parse_time_open_ended_after() {
    let t = jio::parse_time_with_ref("2024年3月之后", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.definition, "blur");
    // End-of-March 2024.
    assert_eq!(t.start.format("%Y-%m").to_string(), "2024-03");
    // Sentinel far-future.
    assert_eq!(t.end.to_string(), "9999-12-31 23:59:59");
}

#[test]
fn parity_parse_time_open_ended_before() {
    // Python: X之前 with past X → end = X's full-day end.
    let t = jio::parse_time_with_ref("2024年春节之前", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.start.to_string(), "0001-01-01 00:00:00");
    assert_eq!(t.end.to_string(), "2024-02-10 23:59:59");
}

// ───────────────────────── parse_location_full — round 21 cases ───────────
// Sourced from jionlp/test/test_location_parser.py.

#[test]
fn parity_parse_location_change2new_county() {
    ensure_init();
    // 柳州地区忻城县 was moved to 来宾市 — change2new=true remaps.
    let r = jio::parse_location_full("柳州地区忻城县", false, true).unwrap();
    assert_eq!(r.province.as_deref(), Some("广西壮族自治区"));
    assert_eq!(r.city.as_deref(), Some("来宾市"));
    assert_eq!(r.county.as_deref(), Some("忻城县"));
    assert_eq!(r.detail, "");
}

#[test]
fn parity_parse_location_change2new_city() {
    ensure_init();
    // 襄樊市 → 襄阳市 (city-level rename).
    let r = jio::parse_location_full("湖北省襄樊市小水街222号", false, true).unwrap();
    assert_eq!(r.province.as_deref(), Some("湖北省"));
    assert_eq!(r.city.as_deref(), Some("襄阳市"));
    assert_eq!(r.detail, "小水街222号");
}

#[test]
fn parity_parse_location_province_city() {
    ensure_init();
    let r = jio::parse_location_full("台湾省台北市", false, true).unwrap();
    assert_eq!(r.province.as_deref(), Some("台湾省"));
    assert_eq!(r.city.as_deref(), Some("台北市"));
    assert_eq!(r.county, None);
    assert_eq!(r.detail, "");
}

#[test]
fn parity_parse_location_province_abbrev() {
    ensure_init();
    // 青海 + 西宁 — the adjacent-offset guard rejects the bad 海西 match.
    let r = jio::parse_location_full("青海西宁", false, false).unwrap();
    assert_eq!(r.province.as_deref(), Some("青海省"));
    assert_eq!(r.city.as_deref(), Some("西宁市"));
    assert_eq!(r.county, None);
    assert_eq!(r.detail, "");
}

#[test]
fn parity_parse_location_alias_exception_suffix() {
    ensure_init();
    // 重庆路 should NOT trigger the 重庆 municipality alias.
    let r = jio::parse_location_full("北海市重庆路其仓11号", false, false).unwrap();
    assert_eq!(r.province.as_deref(), Some("广西壮族自治区"));
    assert_eq!(r.city.as_deref(), Some("北海市"));
    assert_eq!(r.detail, "重庆路其仓11号");
}

#[test]
fn parity_parse_location_autonomous_prefecture() {
    ensure_init();
    let r = jio::parse_location_full("海南藏族自治州", false, false).unwrap();
    assert_eq!(r.province.as_deref(), Some("青海省"));
    assert_eq!(r.city.as_deref(), Some("海南藏族自治州"));
}

#[test]
fn parity_parse_location_alias_ambiguity() {
    ensure_init();
    // 西安 should match 西安市 (city), not 西安区 in 吉林 (county).
    let r = jio::parse_location_full("西安交通大学", false, false).unwrap();
    assert_eq!(r.province.as_deref(), Some("陕西省"));
    assert_eq!(r.city.as_deref(), Some("西安市"));
    assert_eq!(r.detail, "交通大学");
}

#[test]
fn parity_parse_location_economic_zone() {
    ensure_init();
    // 经济技术开发区 normalized to just the suffix.
    let r = jio::parse_location_full("河北省秦皇岛市经济技术开发区", false, false).unwrap();
    assert_eq!(r.province.as_deref(), Some("河北省"));
    assert_eq!(r.city.as_deref(), Some("秦皇岛市"));
    assert_eq!(r.county.as_deref(), Some("经济技术开发区"));
    assert_eq!(r.detail, "");
}

#[test]
fn parity_parse_location_dedupe_city_county() {
    ensure_init();
    // 湖南省长沙市 — 长沙市 (city) full-name must win over 长沙 (county alias).
    let r = jio::parse_location_full("湖南省长沙市", false, false).unwrap();
    assert_eq!(r.province.as_deref(), Some("湖南省"));
    assert_eq!(r.city.as_deref(), Some("长沙市"));
    assert_eq!(r.county, None);
}

#[test]
fn parity_parse_location_full_address() {
    ensure_init();
    let r = jio::parse_location_full("山西长治潞州区山禾路2号", false, false).unwrap();
    assert_eq!(r.province.as_deref(), Some("山西省"));
    assert_eq!(r.city.as_deref(), Some("长治市"));
    assert_eq!(r.county.as_deref(), Some("潞州区"));
    assert_eq!(r.detail, "山禾路2号");
    assert_eq!(r.full_location, "山西省长治市潞州区山禾路2号");
    assert_eq!(r.orig_location, "山西长治潞州区山禾路2号");
}

#[test]
fn parity_parse_location_municipality() {
    ensure_init();
    let r = jio::parse_location_full("重庆解放碑", false, false).unwrap();
    assert_eq!(r.province.as_deref(), Some("重庆市"));
    assert_eq!(r.city.as_deref(), Some("重庆市"));
    assert_eq!(r.detail, "解放碑");
}

#[test]
fn parity_parse_location_nested() {
    ensure_init();
    let r = jio::parse_location_full("湖南湘潭市湘潭县城塘社区", false, false).unwrap();
    assert_eq!(r.province.as_deref(), Some("湖南省"));
    assert_eq!(r.city.as_deref(), Some("湘潭市"));
    assert_eq!(r.county.as_deref(), Some("湘潭县"));
    assert_eq!(r.detail, "城塘社区");
}

#[test]
fn parity_parse_location_no_match_preserves_text() {
    ensure_init();
    let r = jio::parse_location_full("hello world", false, false).unwrap();
    assert_eq!(r.province, None);
    assert_eq!(r.city, None);
    assert_eq!(r.county, None);
    assert_eq!(r.detail, "hello world");
    assert_eq!(r.full_location, "hello world");
    assert_eq!(r.orig_location, "hello world");
}

// ───────────────────────── round 22 — 4 gadgets ──────────────────────────

#[test]
fn parity_province_alias_full_names() {
    assert_eq!(jio::get_china_province_alias("山西省"), Some("山西"));
    assert_eq!(jio::get_china_province_alias("北京市"), Some("北京"));
    assert_eq!(
        jio::get_china_province_alias("新疆维吾尔自治区"),
        Some("新疆")
    );
}

#[test]
fn parity_city_alias() {
    assert_eq!(
        jio::get_china_city_alias("甘孜藏族自治州", false, false),
        Some("甘孜".to_string())
    );
    assert_eq!(
        jio::get_china_city_alias("锡林郭勒盟", false, false),
        Some("锡林郭勒".to_string())
    );
}

#[test]
fn parity_county_alias_qi() {
    // 科尔沁左翼后旗 → 科左后旗 via the Mongolian 旗 mapping.
    assert_eq!(
        jio::get_china_county_alias("科尔沁左翼后旗", false),
        Some("科左后旗".to_string())
    );
    // Generic 县 → drop 县.
    assert_eq!(
        jio::get_china_county_alias("忻城县", false),
        Some("忻城".to_string())
    );
}

#[test]
fn parity_town_alias() {
    assert_eq!(
        jio::get_china_town_alias("苏店镇"),
        Some("苏店".to_string())
    );
    assert_eq!(
        jio::get_china_town_alias("鼓楼街道"),
        Some("鼓楼".to_string())
    );
}

#[test]
fn parity_solar_to_lunar_roundtrip() {
    // 2024-02-10 = lunar 2024 / 1 / 1 (龙年春节).
    let (y, m, d, leap) =
        jio::solar_to_lunar(chrono::NaiveDate::from_ymd_opt(2024, 2, 10).unwrap()).unwrap();
    assert_eq!((y, m, d, leap), (2024, 1, 1, false));

    // 2023-06-22 = lunar 2023 / 5 / 5 (端午节).
    let (y, m, d, leap) =
        jio::solar_to_lunar(chrono::NaiveDate::from_ymd_opt(2023, 6, 22).unwrap()).unwrap();
    assert_eq!((y, m, d, leap), (2023, 5, 5, false));
}

#[test]
fn parity_idiom_solitaire_deterministic() {
    ensure_init();
    // Seed 42 + 千方百计 should deterministically pick the same successor.
    let game = jio::IdiomSolitaireGame::new(42);
    let next1 = game.next_by_char("千方百计", false).unwrap().unwrap();
    assert!(next1.starts_with('计'));

    let game2 = jio::IdiomSolitaireGame::new(42);
    let next2 = game2.next_by_char("千方百计", false).unwrap().unwrap();
    assert_eq!(next1, next2);
}

// ───────────────────────── round 26 — util + dict ────────────────────────

#[test]
fn parity_util_regex_composers() {
    assert_eq!(jio::bracket("\\d+"), "(\\d+)");
    assert_eq!(jio::bracket_absence("A-Z"), "(A-Z)?");
    assert_eq!(jio::absence("x"), "x?");
    assert_eq!(jio::start_end("abc"), "^abc$");
}

#[test]
fn parity_xiehouyu_loads_and_has_known_entry() {
    ensure_init();
    let x = jio::dict::xiehouyu().unwrap();
    assert!(x.len() > 1000);
    // Sanity: dict should contain the classic 一丈厚的烧饼.
    assert!(x.contains_key("一丈厚的烧饼"));
}

#[test]
fn parity_world_location_continents() {
    ensure_init();
    let w = jio::dict::world_location().unwrap();
    assert!(w.iter().any(|r| r.country == "中国"));
    assert!(w.iter().any(|r| r.continent == "亚洲"));
}

#[test]
fn parity_quantifiers_has_entries() {
    ensure_init();
    let q = jio::dict::quantifiers().unwrap();
    assert!(q.len() > 50);
}

// ───────────────────────── round 29 — parse_time tail ────────────────────

#[test]
fn parity_enum_days() {
    let t = jio::parse_time_with_ref("8月14日、15日、16日", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.start.to_string(), "2024-08-14 00:00:00");
    assert_eq!(t.end.to_string(), "2024-08-16 23:59:59");
}

#[test]
fn parity_limit_year_span_month() {
    let t = jio::parse_time_with_ref("今年前两个季度", r17_ref()).unwrap();
    assert_eq!(t.start.to_string(), "2024-01-01 00:00:00");
    assert_eq!(t.end.to_string(), "2024-06-30 23:59:59");
}

#[test]
fn parity_year_solar_season_boundary() {
    let t = jio::parse_time_with_ref("2021年第1季度初", r17_ref()).unwrap();
    assert_eq!(t.definition, "blur");
    assert_eq!(t.start.to_string(), "2021-01-01 00:00:00");
    assert_eq!(t.end.to_string(), "2021-01-31 23:59:59");
}

#[test]
fn parity_school_break() {
    let t = jio::parse_time_with_ref("2024年暑假", r17_ref()).unwrap();
    assert_eq!(t.start.to_string(), "2024-07-01 00:00:00");
    assert_eq!(t.end.to_string(), "2024-08-31 23:59:59");
}

#[test]
fn parity_limit_year_week() {
    let t = jio::parse_time_with_ref("明年第10周", r17_ref()).unwrap();
    assert_eq!(t.start.to_string(), "2025-03-10 00:00:00");
}

#[test]
fn parity_year_ordinal() {
    let t = jio::parse_time_with_ref("第一年", r17_ref()).unwrap();
    assert_eq!(t.start.to_string(), "2024-01-01 00:00:00");
    assert_eq!(t.end.to_string(), "2024-12-31 23:59:59");
}

#[test]
fn parity_yinian_siji() {
    let t = jio::parse_time_with_ref("一年四季", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_delta");
    match t.delta.as_ref().unwrap().year {
        Some(jio::DeltaValue::Single(n)) => assert_eq!(n, 1.0),
        _ => panic!("expected year Single(1)"),
    }
}

#[test]
fn parity_recurring_weekday_filtered() {
    let t = jio::parse_time_with_ref("每周工作日", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_period");
    // now = Fri 2024-03-15; next weekday = Mon 2024-03-18.
    assert_eq!(t.start.to_string(), "2024-03-18 00:00:00");
    match t.delta.as_ref().unwrap().day {
        Some(jio::DeltaValue::Single(n)) => assert_eq!(n, 1.0),
        _ => panic!("expected day Single(1)"),
    }
}

// ───────────────────────── round 32 — partial upgrades ───────────────────

#[test]
fn parity_delta_month_zhihou_upgraded_to_span() {
    // Python: `3个月之后` → open-ended time_span from (now+3mo).
    let t = jio::parse_time_with_ref("3个月之后", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_span");
    // Start lands in June 2024 (~3*30.417 days after now).
    assert_eq!(t.start.format("%Y-%m").to_string(), "2024-06");
}

#[test]
fn parity_delta_hour_yiqian_upgraded_to_span() {
    // Python: `48小时之前` → time_span [now-48h, now].
    let t = jio::parse_time_with_ref("48小时之前", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.start.to_string(), "2024-03-13 10:30:00");
    assert_eq!(t.end.to_string(), "2024-03-15 10:30:00");
}

#[test]
fn parity_delta_quarter_future_point() {
    // Python `两个季度后` = month-span of cur (now + 2 quarters = Sep 2024).
    let t = jio::parse_time_with_ref("两个季度后", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.start.to_string(), "2024-09-01 00:00:00");
    assert_eq!(t.end.to_string(), "2024-09-30 23:59:59");
}

#[test]
fn parity_clock_er_ke() {
    // 二刻 / 两刻 = 30 min.
    let t = jio::parse_time_with_ref("9点二刻", r17_ref()).unwrap();
    assert_eq!(t.start.format("%H:%M").to_string(), "09:30");
    let t = jio::parse_time_with_ref("9点两刻", r17_ref()).unwrap();
    assert_eq!(t.start.format("%H:%M").to_string(), "09:30");
}

#[test]
fn parity_approx_clock_range() {
    // 大约晚上8到10点 — approximation prefix + clock range → blur span.
    let t = jio::parse_time_with_ref("大约晚上8到10点", r17_ref()).unwrap();
    assert_eq!(t.definition, "blur");
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.start.format("%H:%M").to_string(), "20:00");
    assert_eq!(t.end.format("%H:%M").to_string(), "22:00");
}

#[test]
fn parity_blur_year_before() {
    // Python: `32年前` → time_span covering the target year (1992).
    let t = jio::parse_time_with_ref("32年前", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.start.to_string(), "1992-01-01 00:00:00");
    assert_eq!(t.end.to_string(), "1992-12-31 23:59:59");
}

#[test]
fn parity_date_range_tilde() {
    // Date range separated by ASCII tilde.
    let t = jio::parse_time_with_ref("2024年3月~2024年4月", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.start.to_string(), "2024-03-01 00:00:00");
    assert_eq!(t.end.to_string(), "2024-04-30 23:59:59");
}

#[test]
fn parity_nested_delta_quarter_minute() {
    // `一个季度的十五分后` = now + 3 months + 15 min.
    let t = jio::parse_time_with_ref("一个季度的十五分后", r17_ref()).unwrap();
    assert_eq!(t.time_type, "time_point");
    assert_eq!(t.start.to_string(), "2024-06-15 10:45:00");
}

// ───────────────────────── round 33 — parse_money from test_money_parser.py

fn money_case(text: &str, num: f64, case: &str, def: &str) {
    let m = jio::parse_money(text).unwrap_or_else(|| panic!("parse_money({}) returned None", text));
    assert!(
        (m.num - num).abs() < 1e-3,
        "num: got {} vs expected {} ({})",
        m.num,
        num,
        text
    );
    assert_eq!(m.case, case, "case ({})", text);
    assert_eq!(m.definition, def, "definition ({})", text);
}

#[test]
fn parity_money_fullwidth_comma() {
    money_case("82，225.00元", 82225.00, "元", "accurate");
}

#[test]
fn parity_money_hkd() {
    money_case("25481港元", 25481.00, "港元", "accurate");
}

#[test]
fn parity_money_usd_decimal() {
    money_case("45564.44美元", 45564.44, "美元", "accurate");
}

#[test]
fn parity_money_cn_hybrid() {
    money_case("1.2万元", 12000.0, "元", "accurate");
}

#[test]
fn parity_money_quadrillion_jpy() {
    // 3千万亿 = 3 × 10^3 × 10^12 = 3e15.
    money_case("3千万亿日元", 3e15, "日元", "accurate");
}

#[test]
fn parity_money_blur_approx_usd() {
    money_case("约4.287亿美元", 428_700_000.0, "美元", "blur");
}

#[test]
fn parity_money_blur_jin() {
    money_case("近700万元", 7_000_000.0, "元", "blur-");
}

#[test]
fn parity_money_range() {
    let m = jio::parse_money("1万-5万元").unwrap();
    assert!((m.num - 10_000.0).abs() < 1e-3);
    assert_eq!(m.end_num, Some(50_000.0));
    assert_eq!(m.definition, "blur");
}

#[test]
fn parity_money_duo_suffix() {
    money_case("3000多欧元", 3000.0, "欧元", "blur");
}

#[test]
fn parity_money_quantifier_shiduo() {
    let m = jio::parse_money("十几块钱").unwrap();
    assert!((m.num - 10.0).abs() < 1e-3);
    assert_eq!(m.end_num, Some(20.0));
    assert_eq!(m.case, "元");
    assert_eq!(m.definition, "blur");
}

#[test]
fn parity_money_quantifier_jishiwan() {
    // Round 36: Python convention for 几十 = 10..100 (not 10..90).
    let m = jio::parse_money("几十万块").unwrap();
    assert!((m.num - 100_000.0).abs() < 1e-3);
    assert_eq!(m.end_num, Some(1_000_000.0));
    assert_eq!(m.case, "元");
    assert_eq!(m.definition, "blur");
}

#[test]
fn parity_money_consecutive_cn_digits() {
    let m = jio::parse_money("八九亿韩元").unwrap();
    assert!((m.num - 800_000_000.0).abs() < 1e-3);
    assert_eq!(m.end_num, Some(900_000_000.0));
    assert_eq!(m.case, "韩元");
    assert_eq!(m.definition, "blur");
}

// ───────────────────────── Round 36 — all 54 Python test_money_parser cases

/// Assert helper with optional end_num.
fn money_case_full(input: &str, lo: f64, hi: Option<f64>, case: &str, def: &str) {
    let m = jio::parse_money(input).unwrap_or_else(|| panic!("parse_money({}) → None", input));
    assert!(
        (m.num - lo).abs() < 1e-3,
        "{} num: got {}, want {}",
        input,
        m.num,
        lo
    );
    match hi {
        Some(h) => {
            let got = m.end_num.unwrap_or(f64::NAN);
            assert!(
                (got - h).abs() < 1e-3,
                "{} end_num: got {:?}, want {}",
                input,
                m.end_num,
                h
            );
        }
        None => assert_eq!(m.end_num, None, "{} end_num", input),
    }
    assert_eq!(m.case, case, "{} case", input);
    assert_eq!(m.definition, def, "{} definition", input);
}

// ───────────────────────── round 37 — extract_time / extract_money ─────

#[test]
fn parity_extract_time_from_text() {
    ensure_init();
    let now =
        chrono::NaiveDateTime::parse_from_str("2021-09-01T15:15:32", "%Y-%m-%dT%H:%M:%S").unwrap();
    let text = "中秋、国庆两个假期已在眼前。2021年中秋节是9月21日，星期二。";
    let r = jio::extract_time(text, now, false, false);
    let texts: Vec<&str> = r.iter().map(|e| e.text.as_str()).collect();
    assert!(
        texts
            .iter()
            .any(|t| t.contains("2021") || t.contains("中秋") || t.contains("9月")),
        "expected time entity, got {:?}",
        texts
    );
}

#[test]
fn parity_extract_money_from_text() {
    let text = "海航亏损7000万港元出售香港公寓。以2.6亿港元的价格出售";
    let r = jio::extract_money(text, false, false);
    let texts: Vec<&str> = r.iter().map(|e| e.text.as_str()).collect();
    assert!(
        texts.iter().any(|t| t.contains("7000万")),
        "expected 7000万 in {:?}",
        texts
    );
    assert!(
        texts.iter().any(|t| t.contains("2.6亿")),
        "expected 2.6亿 in {:?}",
        texts
    );
}

#[test]
fn parity_money_all_54_from_python() {
    let cases: &[(&str, f64, Option<f64>, &str, &str)] = &[
        ("82，225.00元", 82225.00, None, "元", "accurate"),
        ("25481港元", 25481.00, None, "港元", "accurate"),
        ("45564.44美元", 45564.44, None, "美元", "accurate"),
        (
            "233,333，333,434.344元",
            233333333434.34,
            None,
            "元",
            "accurate",
        ),
        ("1.2万元", 12000.0, None, "元", "accurate"),
        ("3千万亿日元", 3e15, None, "日元", "accurate"),
        (
            "新台币 177.1 亿元",
            17710000000.0,
            None,
            "新台币",
            "accurate",
        ),
        ("15k左右", 15000.0, None, "元", "blur"),
        ("30w上下", 300000.0, None, "元", "blur"),
        ("123元1角1分", 123.11, None, "元", "accurate"),
        (
            "六十四万零一百四十三元一角七分",
            640143.17,
            None,
            "元",
            "accurate",
        ),
        ("壹万二千三百四十五元", 12345.0, None, "元", "accurate"),
        ("三百万", 3000000.0, None, "元", "accurate"),
        ("肆佰叁拾萬", 4300000.0, None, "元", "accurate"),
        ("肆佰叁拾萬圆整", 4300000.0, None, "元", "accurate"),
        ("肆佰叁拾萬圆", 4300000.0, None, "元", "accurate"),
        ("二十五万三千二百泰铢", 253200.0, None, "泰铢", "accurate"),
        ("两个亿卢布", 200000000.0, None, "卢布", "accurate"),
        ("十块三毛", 10.30, None, "元", "accurate"),
        ("一百三十五块六角七分钱", 135.67, None, "元", "accurate"),
        ("港币两千九百六十元", 2960.0, None, "港元", "accurate"),
        ("一百二十三元1角1分", 123.11, None, "元", "accurate"),
        ("三万元欧元", 30000.0, None, "欧元", "accurate"),
        ("9000元日币", 9000.0, None, "日元", "accurate"),
        ("约4.287亿美元", 428700000.0, None, "美元", "blur"),
        ("近700万元", 7000000.0, None, "元", "blur-"),
        ("至少九千块钱以上", 9000.0, None, "元", "blur+"),
        ("不到1.9万台币", 19000.0, None, "新台币", "blur-"),
        ("小于40万", 400000.0, None, "元", "blur-"),
        ("3000多欧元", 3000.0, Some(4000.0), "欧元", "blur"),
        ("几十万块", 100000.0, Some(1000000.0), "元", "blur"),
        (
            "人民币数十亿元",
            1000000000.0,
            Some(10000000000.0),
            "元",
            "blur",
        ),
        (
            "数十亿元人民币",
            1000000000.0,
            Some(10000000000.0),
            "元",
            "blur",
        ),
        ("十几块钱", 10.0, Some(20.0), "元", "blur"),
        ("大约十多欧元", 10.0, Some(20.0), "欧元", "blur"),
        ("从8500到3万港元", 8500.0, Some(30000.0), "港元", "blur"),
        ("1万-5万元", 10000.0, Some(50000.0), "元", "blur"),
        ("1万元--5万元", 10000.0, Some(50000.0), "元", "blur"),
        ("10~15k元", 10000.0, Some(15000.0), "元", "blur"),
        ("2——3万港币", 20000.0, Some(30000.0), "港元", "blur"),
        ("两到三万港元", 20000.0, Some(30000.0), "港元", "blur"),
        ("十八至三十万日元", 180000.0, Some(300000.0), "日元", "blur"),
        ("两到三仟澳元", 2000.0, Some(3000.0), "澳元", "blur"),
        ("两~3百日元", 200.0, Some(300.0), "日元", "blur"),
        (
            "一百二十到一百五十万元",
            1200000.0,
            Some(1500000.0),
            "元",
            "blur",
        ),
        (
            "一千到两千万元人民币",
            10000000.0,
            Some(20000000.0),
            "元",
            "blur",
        ),
        (
            "七千到九千亿元",
            700000000000.0,
            Some(900000000000.0),
            "元",
            "blur",
        ),
        (
            "八到九百亿泰铢",
            800000000.0,
            Some(90000000000.0),
            "泰铢",
            "blur",
        ),
        (
            "八百到九百亿泰铢",
            80000000000.0,
            Some(90000000000.0),
            "泰铢",
            "blur",
        ),
        ("八九亿韩元", 800000000.0, Some(900000000.0), "韩元", "blur"),
        ("三五百块", 300.0, Some(500.0), "元", "blur"),
        ("四五千块钱", 4000.0, Some(5000.0), "元", "blur"),
        ("50万元（含）以上", 500000.0, None, "元", "blur+"),
        ("1万(含)-5万元", 10000.0, Some(50000.0), "元", "blur"),
    ];
    for (input, lo, hi, case, def) in cases {
        money_case_full(input, *lo, *hi, case, def);
    }
}

// ───────────────────────── round 33 — parse_time from test_time_parser.py

fn ts1() -> chrono::NaiveDateTime {
    chrono::NaiveDateTime::parse_from_str("2021-06-14T01:06:40", "%Y-%m-%dT%H:%M:%S").unwrap()
}

#[test]
fn parity_time_py_eight_digit() {
    let t = jio::parse_time_with_ref("20240307", ts1()).unwrap();
    assert_eq!(t.time_type, "time_point");
    assert_eq!(t.start.to_string(), "2024-03-07 00:00:00");
    assert_eq!(t.end.to_string(), "2024-03-07 23:59:59");
}

#[test]
fn parity_time_py_slash_format() {
    let t = jio::parse_time_with_ref("2019/04/19", ts1()).unwrap();
    assert_eq!(t.start.to_string(), "2019-04-19 00:00:00");
}

#[test]
fn parity_time_py_minute_precision() {
    // Minute-precision end goes to :59 second.
    let t = jio::parse_time_with_ref("2018-11-29 18:59", ts1()).unwrap();
    assert_eq!(t.start.to_string(), "2018-11-29 18:59:00");
    assert_eq!(t.end.to_string(), "2018-11-29 18:59:59");
}

#[test]
fn parity_time_py_dot_separated() {
    let t = jio::parse_time_with_ref("2019.9.6", ts1()).unwrap();
    assert_eq!(t.start.to_string(), "2019-09-06 00:00:00");
}

#[test]
fn parity_time_py_bare_year() {
    let t = jio::parse_time_with_ref("2018", ts1()).unwrap();
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.start.to_string(), "2018-01-01 00:00:00");
    assert_eq!(t.end.to_string(), "2018-12-31 23:59:59");
}

#[test]
fn parity_time_py_two_digit_year() {
    let t = jio::parse_time_with_ref("03年2月28日", ts1()).unwrap();
    assert_eq!(t.start.to_string(), "2003-02-28 00:00:00");
}

#[test]
fn parity_time_py_month_only() {
    let t = jio::parse_time_with_ref("98年4月", ts1()).unwrap();
    assert_eq!(t.start.to_string(), "1998-04-01 00:00:00");
    assert_eq!(t.end.to_string(), "1998-04-30 23:59:59");
}

// ───────────────────────── round 33 — remove_* / idiom / text_aug ────────

#[test]
fn parity_remove_email_basic() {
    let out = jio::remove_email("联系 dongrixinyu.89@163.com 请速回");
    assert!(!out.contains("@"));
    assert!(out.contains("联系"));
    assert!(out.contains("请速回"));
}

#[test]
fn parity_remove_url_basic() {
    let out = jio::remove_url("抖音 https://v.douyin.com/RtKFFah/ 看一下");
    assert!(!out.contains("http"));
    assert!(out.contains("抖音"));
    assert!(out.contains("看一下"));
}

#[test]
fn parity_remove_phone_basic() {
    let out = jio::remove_phone_number("打电话到 13812345678 咨询");
    assert!(!out.contains("13812345678"));
    assert!(out.contains("打电话到"));
}

#[test]
fn parity_remove_email_with_prefix() {
    // Python: remove_email(text, delete_prefix=True) removes `E-mail: ` too.
    let out = jio::remove_email_with_prefix("请联系 E-mail: alice@ex.com 或 邮箱: bob@ex.com 都行");
    assert!(!out.contains("alice"));
    assert!(!out.contains("bob"));
    assert!(!out.contains("E-mail"));
    assert!(!out.contains("邮箱"));
}

#[test]
fn parity_remove_url_with_prefix() {
    let out = jio::remove_url_with_prefix("详情 网址: https://example.com 查看");
    assert!(!out.contains("https"));
    assert!(!out.contains("网址"));
    assert!(out.contains("详情"));
}

#[test]
fn parity_remove_phone_with_prefix() {
    let out = jio::remove_phone_number_with_prefix("电话: 13812345678 联系");
    assert!(!out.contains("13812345678"));
    assert!(!out.contains("电话"));
}

// ───────────────────────── round 43 — final 3 files ──────────────────────

#[test]
fn parity_time_period_parser() {
    // normalize_time_period: `两年` → {year: 2}.
    let d = jio::normalize_time_period("两年").unwrap();
    match d.year {
        Some(jio::DeltaValue::Single(n)) => assert_eq!(n, 2.0),
        _ => panic!("expected year=2"),
    }
}

#[test]
fn parity_analyse_ner_dataset_split() {
    let x: Vec<String> = (0..100).map(|i| format!("s{}", i)).collect();
    let y: Vec<Vec<jio::Entity>> = (0..100)
        .map(|i| {
            vec![jio::Entity {
                text: "x".into(),
                type_: if i % 2 == 0 { "A".into() } else { "B".into() },
                offset: (0, 1),
            }]
        })
        .collect();
    let r = jio::analyse_ner_dataset_split(&x, &y, (0.8, 0.1, 0.1), 42, true);
    assert_eq!(r.train_x.len(), 80);
    assert_eq!(r.valid_x.len(), 10);
    assert_eq!(r.test_x.len(), 10);
    assert!(r.stats.total.contains_key("A") && r.stats.total.contains_key("B"));
}

#[test]
fn parity_rule_mining_context_capture() {
    let texts = ["公司: 百度", "公司: 腾讯"];
    let entities = vec![
        vec![("Company".to_string(), 4, 6)],
        vec![("Company".to_string(), 4, 6)],
    ];
    let r = jio::mine_rules(&texts, &entities, 4);
    let comp = r.get("Company").unwrap();
    assert!(!comp.prefix_freq.is_empty());
}

#[test]
fn parity_idiom_solitaire_empty_input() {
    // Python: `jio.idiom_solitaire('', ...)` returns ''. Our helper treats
    // empty cur as missing last char and returns Ok(None) → we surface None.
    let game = jio::IdiomSolitaireGame::new(1);
    let r = game.next_by_char("", false);
    assert!(r.is_ok(), "empty input should not panic");
}

#[test]
fn parity_idiom_solitaire_daozuqichang() {
    ensure_init();
    // Python: 道阻且长 → idiom starting with 长.
    let game = jio::IdiomSolitaireGame::new(7);
    let n = game.next_by_char("道阻且长", false).unwrap().unwrap();
    assert!(n.starts_with('长'), "expected 长-prefix idiom, got {}", n);
}

#[test]
fn parity_text_aug_swap_runs() {
    // test_text_aug.py verifies the augmenter runs and produces outputs.
    let r = jio::swap_char_position("今天天气真好", 2, 0.3, 12345, 0.5);
    assert_eq!(r.len(), 2);
    for out in &r {
        assert_eq!(out.chars().count(), "今天天气真好".chars().count());
    }
}

#[test]
fn parity_text_aug_homophone() {
    ensure_init();
    let r = jio::homophone_substitution("今天天气真好", 2, 0.5, 12345).unwrap();
    assert_eq!(r.len(), 2);
    for out in &r {
        assert_eq!(out.chars().count(), "今天天气真好".chars().count());
    }
}

// ───────────────────────── round 34 — location town_village 5-level ──────

#[test]
fn parity_location_town_village() {
    ensure_init();
    // Python case: `老河口市天气` + town_village=true → town=None
    // (Python dict may have "老河口市" mapped under 湖北省襄阳市 with town
    // "城关镇" etc., but detail="天气" has no town substring, so town=None).
    let r = jio::parse_location_full("老河口市天气", true, true).unwrap();
    assert_eq!(r.province.as_deref(), Some("湖北省"));
    assert_eq!(r.county.as_deref(), Some("老河口市"));
    // Detail has no matching town — so both fields are None.
    assert_eq!(r.town, None);
    assert_eq!(r.village, None);
}

#[test]
fn parity_time_py_double_em_dash() {
    // Pattern #105 — `——` (double em-dash) as range separator.
    let t = jio::parse_time_with_ref("2024年3月——2024年4月", ts1()).unwrap();
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.start.to_string(), "2024-03-01 00:00:00");
    assert_eq!(t.end.to_string(), "2024-04-30 23:59:59");
}

#[test]
fn parity_time_py_double_dash_ascii() {
    let t = jio::parse_time_with_ref("2024年3月--2024年4月", ts1()).unwrap();
    assert_eq!(t.start.to_string(), "2024-03-01 00:00:00");
}

#[test]
fn parity_time_py_ym_range_mixed() {
    // Python: `1999.08-2002.02` → time_span 1999-08-01..2002-02-28.
    let t = jio::parse_time_with_ref("1999.08-2002.02", ts1()).unwrap();
    assert_eq!(t.time_type, "time_span");
    assert_eq!(t.start.to_string(), "1999-08-01 00:00:00");
    assert_eq!(t.end.to_string(), "2002-02-28 23:59:59");
}

#[test]
fn parity_time_py_ym_to_year() {
    // Python: `2008.03-2009` → time_span 2008-03-01..2009-12-31.
    let t = jio::parse_time_with_ref("2008.03-2009", ts1()).unwrap();
    assert_eq!(t.start.to_string(), "2008-03-01 00:00:00");
    assert_eq!(t.end.to_string(), "2009-12-31 23:59:59");
}

#[test]
fn parity_time_py_fullwidth_colon() {
    let t = jio::parse_time_with_ref("2021-09-12-11：23", ts1()).unwrap();
    assert_eq!(t.start.to_string(), "2021-09-12 11:23:00");
    assert_eq!(t.end.to_string(), "2021-09-12 11:23:59");
}

#[test]
fn parity_location_town_village_matched() {
    ensure_init();
    // Python test case:
    //   云南省红河哈尼族彝族自治州元阳县黄茅岭乡 → town=黄茅岭乡
    let r =
        jio::parse_location_full("云南省红河哈尼族彝族自治州元阳县黄茅岭乡", true, true).unwrap();
    assert_eq!(r.province.as_deref(), Some("云南省"));
    assert_eq!(r.city.as_deref(), Some("红河哈尼族彝族自治州"));
    assert_eq!(r.county.as_deref(), Some("元阳县"));
    assert_eq!(r.town.as_deref(), Some("黄茅岭乡"));
}

// ───────────────────────── round 37 — all Python test_location_parser cases

/// Fields we assert on for each Python location parity case. `None`
/// slots in `province/city/county` mean the Python expected-null entry;
/// `town`/`village` fields are only checked when town_village=true.
struct LocExpect<'a> {
    input: &'a str,
    town_village: bool,
    change2new: bool,
    province: Option<&'a str>,
    city: Option<&'a str>,
    county: Option<&'a str>,
    detail: &'a str,
    town: Option<&'a str>,
    village: Option<&'a str>,
}

fn loc_case(e: &LocExpect) {
    let r = jio::parse_location_full(e.input, e.town_village, e.change2new).unwrap();
    assert_eq!(
        r.province.as_deref(),
        e.province,
        "{} province: got {:?} want {:?}",
        e.input,
        r.province,
        e.province
    );
    assert_eq!(
        r.city.as_deref(),
        e.city,
        "{} city: got {:?} want {:?}",
        e.input,
        r.city,
        e.city
    );
    assert_eq!(
        r.county.as_deref(),
        e.county,
        "{} county: got {:?} want {:?}",
        e.input,
        r.county,
        e.county
    );
    assert_eq!(
        r.detail, e.detail,
        "{} detail: got {:?} want {:?}",
        e.input, r.detail, e.detail
    );
    if e.town_village {
        assert_eq!(
            r.town.as_deref(),
            e.town,
            "{} town: got {:?} want {:?}",
            e.input,
            r.town,
            e.town
        );
        assert_eq!(
            r.village.as_deref(),
            e.village,
            "{} village: got {:?} want {:?}",
            e.input,
            r.village,
            e.village
        );
    }
}

#[test]
fn parity_location_all_from_python() {
    ensure_init();
    use LocExpect as L;
    let cases: &[LocExpect] = &[
        L {
            input: "柳州地区忻城县",
            town_village: false,
            change2new: true,
            province: Some("广西壮族自治区"),
            city: Some("来宾市"),
            county: Some("忻城县"),
            detail: "",
            town: None,
            village: None,
        },
        L {
            input: "台湾省台北市",
            town_village: false,
            change2new: true,
            province: Some("台湾省"),
            city: Some("台北市"),
            county: None,
            detail: "",
            town: None,
            village: None,
        },
        L {
            input: "湖北省襄樊市小水街222号",
            town_village: false,
            change2new: true,
            province: Some("湖北省"),
            city: Some("襄阳市"),
            county: None,
            detail: "小水街222号",
            town: None,
            village: None,
        },
        L {
            input: "老河口市天气",
            town_village: true,
            change2new: true,
            province: Some("湖北省"),
            city: Some("襄阳市"),
            county: Some("老河口市"),
            detail: "天气",
            town: None,
            village: None,
        },
        L {
            input: "河北区",
            town_village: true,
            change2new: true,
            province: Some("天津市"),
            city: Some("天津市"),
            county: Some("河北区"),
            detail: "",
            town: None,
            village: None,
        },
        L {
            input: "湘潭城塘社区",
            town_village: true,
            change2new: true,
            province: Some("湖南省"),
            city: Some("湘潭市"),
            county: None,
            detail: "城塘社区",
            town: None,
            village: None,
        },
        L {
            input: "湘潭县城塘社区",
            town_village: true,
            change2new: true,
            province: Some("湖南省"),
            city: Some("湘潭市"),
            county: Some("湘潭县"),
            detail: "城塘社区",
            town: None,
            village: None,
        },
        L {
            input: "云南省红河哈尼族彝族自治州元阳县黄茅岭乡",
            town_village: true,
            change2new: true,
            province: Some("云南省"),
            city: Some("红河哈尼族彝族自治州"),
            county: Some("元阳县"),
            detail: "黄茅岭乡",
            town: Some("黄茅岭乡"),
            village: None,
        },
        L {
            input: "吉林省吉林市小皇村",
            town_village: true,
            change2new: true,
            province: Some("吉林省"),
            city: Some("吉林市"),
            county: None,
            detail: "小皇村",
            town: None,
            village: None,
        },
        L {
            input: "重庆解放碑",
            town_village: true,
            change2new: true,
            province: Some("重庆市"),
            city: Some("重庆市"),
            county: None,
            detail: "解放碑",
            town: None,
            village: None,
        },
        L {
            input: "湖南湘潭城塘社区",
            town_village: true,
            change2new: true,
            province: Some("湖南省"),
            city: Some("湘潭市"),
            county: None,
            detail: "城塘社区",
            town: None,
            village: None,
        },
        L {
            input: "湖南湘潭市湘潭县城塘社区",
            town_village: true,
            change2new: true,
            province: Some("湖南省"),
            city: Some("湘潭市"),
            county: Some("湘潭县"),
            detail: "城塘社区",
            town: None,
            village: None,
        },
        L {
            input: "西湖区蒋村花园小区管局农贸市场",
            town_village: true,
            change2new: true,
            province: None,
            city: None,
            county: Some("西湖区"),
            detail: "蒋村花园小区管局农贸市场",
            town: None,
            village: None,
        },
        L {
            input: "山西长治潞州区山禾路2号",
            town_village: false,
            change2new: false,
            province: Some("山西省"),
            city: Some("长治市"),
            county: Some("潞州区"),
            detail: "山禾路2号",
            town: None,
            village: None,
        },
        L {
            input: "青海西宁",
            town_village: false,
            change2new: false,
            province: Some("青海省"),
            city: Some("西宁市"),
            county: None,
            detail: "",
            town: None,
            village: None,
        },
        L {
            input: "东兴市北仑大道59号",
            town_village: true,
            change2new: false,
            province: Some("广西壮族自治区"),
            city: Some("防城港市"),
            county: Some("东兴市"),
            detail: "北仑大道59号",
            town: None,
            village: None,
        },
        L {
            input: "北海市重庆路其仓11号",
            town_village: true,
            change2new: false,
            province: Some("广西壮族自治区"),
            city: Some("北海市"),
            county: None,
            detail: "重庆路其仓11号",
            town: None,
            village: None,
        },
        L {
            input: "海南藏族自治州",
            town_village: true,
            change2new: false,
            province: Some("青海省"),
            city: Some("海南藏族自治州"),
            county: None,
            detail: "",
            town: None,
            village: None,
        },
        L {
            input: "西安交通大学",
            town_village: true,
            change2new: false,
            province: Some("陕西省"),
            city: Some("西安市"),
            county: None,
            detail: "交通大学",
            town: None,
            village: None,
        },
        L {
            input: "河北省秦皇岛市经济技术开发区",
            town_village: true,
            change2new: false,
            province: Some("河北省"),
            city: Some("秦皇岛市"),
            county: Some("经济技术开发区"),
            detail: "",
            town: None,
            village: None,
        },
        L {
            input: "江西南昌市新建区松湖镇江西省南昌市新建区松湖镇松湖中心小学",
            town_village: true,
            change2new: true,
            province: Some("江西省"),
            city: Some("南昌市"),
            county: Some("新建区"),
            detail: "松湖镇江西省南昌市新建区松湖镇松湖中心小学",
            town: Some("松湖镇"),
            village: None,
        },
        L {
            input: "湖南省长沙市",
            town_village: true,
            change2new: false,
            province: Some("湖南省"),
            city: Some("长沙市"),
            county: None,
            detail: "",
            town: None,
            village: None,
        },
        L {
            input: "香港九龙半岛清水湾香港科技大学",
            town_village: false,
            change2new: false,
            province: Some("香港特别行政区"),
            city: Some("香港"),
            county: Some("九龙城区"),
            detail: "半岛清水湾香港科技大学",
            town: None,
            village: None,
        },
        L {
            input: "莱芜",
            town_village: false,
            change2new: false,
            province: Some("山东省"),
            city: Some("济南市"),
            county: Some("莱芜区"),
            detail: "",
            town: None,
            village: None,
        },
        L {
            input: "成都市新津县金华镇清云北路2号（四川新津工业园）",
            town_village: false,
            change2new: false,
            province: Some("四川省"),
            city: Some("成都市"),
            county: Some("新津区"),
            detail: "金华镇清云北路2号（四川新津工业园）",
            town: None,
            village: None,
        },
        L {
            input: "青岛市市南区香港中路18号",
            town_village: false,
            change2new: false,
            province: Some("山东省"),
            city: Some("青岛市"),
            county: Some("市南区"),
            detail: "香港中路18号",
            town: None,
            village: None,
        },
        L {
            input: "石首市笔架山办事处建设路香港城西街",
            town_village: false,
            change2new: false,
            province: Some("湖北省"),
            city: Some("荆州市"),
            county: Some("石首市"),
            detail: "笔架山办事处建设路香港城西街",
            town: None,
            village: None,
        },
        L {
            input: "新疆巴音郭楞",
            town_village: false,
            change2new: false,
            province: Some("新疆维吾尔自治区"),
            city: Some("巴音郭楞蒙古自治州"),
            county: None,
            detail: "",
            town: None,
            village: None,
        },
        L {
            input: "内蒙古自治区通辽市科尔沁左翼后旗甘蓝子街98号",
            town_village: true,
            change2new: true,
            province: Some("内蒙古自治区"),
            city: Some("通辽市"),
            county: Some("科尔沁左翼后旗"),
            detail: "甘蓝子街98号",
            town: None,
            village: None,
        },
        L {
            input: "内蒙古自治区通辽市科尔沁甘蓝子街98号",
            town_village: true,
            change2new: true,
            province: Some("内蒙古自治区"),
            city: Some("通辽市"),
            county: Some("科尔沁区"),
            detail: "甘蓝子街98号",
            town: None,
            village: None,
        },
        L {
            input: "内蒙古通辽市科尔沁左翼后旗",
            town_village: true,
            change2new: true,
            province: Some("内蒙古自治区"),
            city: Some("通辽市"),
            county: Some("科尔沁左翼后旗"),
            detail: "",
            town: None,
            village: None,
        },
        L {
            input: "库尔勒市祥和镇",
            town_village: false,
            change2new: true,
            province: Some("新疆维吾尔自治区"),
            city: Some("巴音郭楞蒙古自治州"),
            county: Some("库尔勒市"),
            detail: "祥和镇",
            town: None,
            village: None,
        },
        L {
            input: "台湾省屏东县屏东市太原1路84号",
            town_village: false,
            change2new: true,
            province: Some("台湾省"),
            city: None,
            county: Some("屏东县"),
            detail: "屏东市太原1路84号",
            town: None,
            village: None,
        },
    ];
    for c in cases {
        loc_case(c);
    }
}
