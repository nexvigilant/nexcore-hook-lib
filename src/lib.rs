//! Minimal hook utilities for Claude Code hooks.
//!
//! # Codex Compliance
//!
//! - **Tier**: T2-C (Cross-Domain Composite)
//! - **Grounding**: Types ground to T1 (String, u8, f64) via T2-P (ToolName, ToolNameCode, Verdict, VerdictCode, Decision, Reason, Confidence).
//! - **Quantification**: Tool names and verdicts are enumerated; decisions derive from verdicts.
//!
//! Hook Protocol:
//! - Input: JSON on stdin with tool_input, tool_name, etc.
//! - Output: JSON on stdout (empty `{}` for pass)
//! - Exit codes: 0 = pass, 1 = warn, 2 = block
//!
//! # Neurotransmitter Pattern
//!
//! For telemetry and metrics, use fire-and-forget signal emission:
//!
//! ```rust,ignore
//! use nexcore_hook_lib::neurotransmitter::{emit, Signal};
//!
//! // Sub-millisecond - emits signal, doesn't wait
//! emit(Signal::new("skill_invoked").with_data("skill", "extractor"));
//! ```

#![warn(missing_docs)]
pub mod atomic;
pub mod neurotransmitter;

/// Cytokine bridge for Guardian integration.
///
/// Enables hooks to emit typed signals to the Guardian homeostasis loop.
/// Requires the `cytokine` feature flag.
///
/// # Example
/// ```rust,ignore
/// use nexcore_hook_lib::cytokine::{emit_tool_blocked, emit_hook_completed};
///
/// // Tool blocked by hook → TNF-alpha
/// emit_tool_blocked("Write", "unwrap-guardian", "unwrap() detected");
///
/// // Hook completed → TGF-beta (regulation)
/// emit_hook_completed("my-hook", 1500, "pass");
/// ```
#[cfg(feature = "cytokine")]
pub mod cytokine;

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::io::{self, Read};
use std::path::Path;
use std::process;

pub const DEFAULT_SNIPPET_LEN: usize = 60;
pub const DEFAULT_MAX_VIOLATIONS: usize = 5;
pub const MAX_CONTENT_BYTES: usize = 200_000;
pub const MAX_SCAN_LINES: usize = 5_000;

const ENV_SNIPPET_LEN: &str = "CODEX_HOOK_SNIPPET_LEN";
const ENV_MAX_VIOLATIONS: &str = "CODEX_HOOK_MAX_VIOLATIONS";
const ENV_MAX_CONTENT_BYTES: &str = "CODEX_HOOK_MAX_CONTENT_BYTES";
const ENV_MAX_SCAN_LINES: &str = "CODEX_HOOK_MAX_SCAN_LINES";

// Limits can be overridden via the CODEX_HOOK_* environment variables above.

/// Enumeration of known tools.
///
/// # Tier: T2-P
/// Grounds to: T1(String) via serde string representation.
/// Repr: not applied due to data-carrying Unknown(String) variant; quantification via ToolNameCode.
/// Ord: not implemented (no natural ordering for tool identities).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolName {
    Edit,
    Replace,
    WriteFile,
    ReadFile,
    RunShellCommand,
    ListDirectory,
    Glob,
    SearchFileContent,
    #[serde(untagged)]
    Unknown(String),
}

impl fmt::Display for ToolName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Edit => write!(f, "edit"),
            Self::Replace => write!(f, "replace"),
            Self::WriteFile => write!(f, "write_file"),
            Self::ReadFile => write!(f, "read_file"),
            Self::RunShellCommand => write!(f, "run_shell_command"),
            Self::ListDirectory => write!(f, "list_directory"),
            Self::Glob => write!(f, "glob"),
            Self::SearchFileContent => write!(f, "search_file_content"),
            Self::Unknown(s) => write!(f, "{}", s),
        }
    }
}

/// Tool name quantitative code.
///
/// # Tier: T2-P
/// Grounds to: T1(u64).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolNameCode(pub u64);

impl From<&ToolName> for ToolNameCode {
    fn from(value: &ToolName) -> Self {
        let code = match value {
            ToolName::Edit => 1,
            ToolName::Replace => 2,
            ToolName::WriteFile => 3,
            ToolName::ReadFile => 4,
            ToolName::RunShellCommand => 5,
            ToolName::ListDirectory => 6,
            ToolName::Glob => 7,
            ToolName::SearchFileContent => 8,
            ToolName::Unknown(name) => fnv1a_64(name.as_bytes()),
        };
        Self(code)
    }
}

impl From<ToolName> for ToolNameCode {
    fn from(value: ToolName) -> Self {
        Self::from(&value)
    }
}

/// Hook input from Claude Code.
///
/// # Tier: T2-C
/// Composite input structure.
/// Grounds to: T1(String) via Option, plus T2-P(ToolName) -> T1(String) and T2-C(ToolInput) -> T1(String).
#[derive(Debug, Deserialize)]
pub struct HookInput {
    pub session_id: Option<String>,
    pub cwd: Option<String>,
    pub tool_name: Option<ToolName>,
    pub tool_input: Option<ToolInput>,
}

/// Tool-specific input data.
///
/// # Tier: T2-C
/// Composite tool arguments.
/// Grounds to: T1(String) via Option.
#[derive(Debug, Deserialize)]
pub struct ToolInput {
    pub file_path: Option<String>,
    pub content: Option<String>,
    pub command: Option<String>,
    pub old_string: Option<String>,
    pub new_string: Option<String>,
    pub timeout: Option<u64>,
    pub description: Option<String>,
    pub run_in_background: Option<bool>,
    pub max_turns: Option<u64>,
    pub subagent_type: Option<String>,
    pub prompt: Option<String>,
    pub model: Option<String>,
}

/// Hook decision verdict.
///
/// # Tier: T2-P
/// Grounds to: T1(u8) via repr.
/// Ord: Pass < Warn < Block.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Pass = 0,
    Warn = 1,
    Block = 2,
}

/// Verdict quantitative code.
///
/// # Tier: T2-P
/// Grounds to: T1(u8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct VerdictCode(pub u8);

impl From<Verdict> for VerdictCode {
    fn from(value: Verdict) -> Self {
        Self(value as u8)
    }
}

/// Hook decision string for output.
///
/// # Tier: T2-P
/// Grounds to: T1(String).
/// Ord: not implemented (semantic string, no natural ordering).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Decision(pub String);

impl From<Verdict> for Decision {
    fn from(verdict: Verdict) -> Self {
        let label = match verdict {
            Verdict::Pass => "pass",
            Verdict::Warn => "warn",
            Verdict::Block => "block",
        };
        Self(label.to_string())
    }
}

/// Hook reason string for output.
///
/// # Tier: T2-P
/// Grounds to: T1(String).
/// Ord: not implemented (semantic string, no natural ordering).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Reason(pub String);

/// Confidence score for computed results.
///
/// # Tier: T2-P
/// Grounds to: T1(f64).
/// Ord: not implemented (f64 NaN makes total ordering undefined).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Confidence(pub f64);

impl Confidence {
    pub const fn certain() -> Self {
        Self(1.0)
    }
}

/// Limited content view for scanning.
///
/// # Tier: T2-C
/// Grounds to: T1(String, bool, u64) via &str and byte counts.
#[derive(Debug, Clone, Copy)]
pub struct ContentSlice<'a> {
    pub text: &'a str,
    pub total_bytes: usize,
    pub scanned_bytes: usize,
    pub truncated: bool,
}

/// Evidence line number.
///
/// # Tier: T2-P
/// Grounds to: T1(u64).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LineNumber(pub u64);

/// Evidence kind label.
///
/// # Tier: T2-P
/// Grounds to: T1(String).
/// Ord: not implemented (semantic string, no natural ordering).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EvidenceKind(pub String);

/// Evidence snippet.
///
/// # Tier: T2-P
/// Grounds to: T1(String).
/// Ord: not implemented (semantic string, no natural ordering).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EvidenceSnippet(pub String);

/// Evidence line for a violation.
///
/// # Tier: T2-C
/// Grounds to: T2-P(LineNumber, EvidenceKind, EvidenceSnippet) -> T1(u64, String).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceLine {
    pub line: LineNumber,
    pub kind: EvidenceKind,
    pub snippet: EvidenceSnippet,
}

impl EvidenceLine {
    pub fn new(line_num: usize, kind: impl Into<String>, snippet: impl Into<String>) -> Self {
        Self {
            line: LineNumber(line_num as u64),
            kind: EvidenceKind(kind.into()),
            snippet: EvidenceSnippet(snippet.into()),
        }
    }
}

/// Violation with explicit confidence.
///
/// # Tier: T2-C
/// Grounds to: T2-C(EvidenceLine) -> T2-P(LineNumber, EvidenceKind, EvidenceSnippet) and T2-P(Confidence).
#[derive(Debug, Clone)]
pub struct Violation {
    pub evidence: EvidenceLine,
    pub confidence: Confidence,
}

impl Violation {
    pub fn new(evidence: EvidenceLine, confidence: Confidence) -> Self {
        Self {
            evidence,
            confidence,
        }
    }
}

/// Format a standard violation header and list.
pub fn format_violations(title: &str, violations: &[Violation]) -> String {
    let mut msg = format!("BLOCKED: {title}\n\n");
    let max = max_violations();
    for violation in violations.iter().take(max) {
        let line = violation.evidence.line.0;
        let kind = &violation.evidence.kind.0;
        let snippet = &violation.evidence.snippet.0;
        let confidence = violation.confidence.0;
        msg.push_str(&format!(
            "  Line {}: {} - `{}` (confidence {:.2})\n",
            line, kind, snippet, confidence
        ));
    }
    if violations.len() > max {
        msg.push_str(&format!("  ... and {} more\n", violations.len() - max));
    }
    msg
}

/// Append scan-limit notes when truncation or line caps are applied.
pub fn append_scan_notice(
    msg: &mut String,
    content: &ContentSlice<'_>,
    truncated_lines: bool,
    truncated_hits: bool,
) {
    if !content.truncated && !truncated_lines && !truncated_hits {
        return;
    }
    msg.push_str("\nScan limits applied:\n");
    if content.truncated {
        msg.push_str(&format!(
            "  - Scanned first {} bytes of {}\n",
            content.scanned_bytes, content.total_bytes
        ));
    }
    if truncated_lines {
        msg.push_str(&format!("  - Scanned first {} lines\n", max_scan_lines()));
    }
    if truncated_hits {
        msg.push_str(&format!(
            "  - Stopped after {} violations\n",
            max_violations()
        ));
    }
}

/// Computed value with explicit confidence.
///
/// # Tier: T2-C
/// Grounds to: T2-P(Confidence) -> T1(f64) and T (caller-grounded).
#[derive(Debug, Clone)]
pub struct Measured<T> {
    pub value: T,
    pub confidence: Confidence,
}

/// Hook output response.
///
/// # Tier: T2-C
/// Composite output structure.
/// Grounds to: T2-P(Decision, Reason) -> T1(String) via Option.
#[derive(Debug, Serialize, Default)]
pub struct HookOutput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision: Option<Decision>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<Reason>,
}

impl HookInput {
    /// Get file_path from tool_input
    pub fn file_path(&self) -> Option<&str> {
        self.tool_input.as_ref()?.file_path.as_deref()
    }

    /// Get working directory
    pub fn cwd(&self) -> Option<&str> {
        self.cwd.as_deref()
    }
}

/// Get file_path or exit with pass if missing.
pub fn file_path_or_pass(input: &HookInput) -> &str {
    match input.file_path() {
        Some(path) => path,
        None => pass(),
    }
}

/// Get content or exit with pass if missing/empty.
pub fn content_or_pass(input: &HookInput) -> &str {
    let content = input
        .tool_input
        .as_ref()
        .and_then(|t| t.new_string.as_deref().or(t.content.as_deref()))
        .unwrap_or("");
    if content.is_empty() {
        pass();
    }
    content
}

/// Get content with size cap or exit with pass if missing/empty.
pub fn content_or_pass_limited(input: &HookInput) -> ContentSlice<'_> {
    let content = content_or_pass(input);
    let total_bytes = content.len();
    let max_bytes = max_content_bytes();
    if total_bytes <= max_bytes {
        return ContentSlice {
            text: content,
            total_bytes,
            scanned_bytes: total_bytes,
            truncated: false,
        };
    }
    let text = slice_to_char_boundary(content, max_bytes);
    ContentSlice {
        text,
        total_bytes,
        scanned_bytes: text.len(),
        truncated: true,
    }
}

pub fn snippet_len() -> usize {
    env_usize(ENV_SNIPPET_LEN, DEFAULT_SNIPPET_LEN)
}

pub fn max_violations() -> usize {
    env_usize(ENV_MAX_VIOLATIONS, DEFAULT_MAX_VIOLATIONS)
}

pub fn max_content_bytes() -> usize {
    env_usize(ENV_MAX_CONTENT_BYTES, MAX_CONTENT_BYTES)
}

pub fn max_scan_lines() -> usize {
    env_usize(ENV_MAX_SCAN_LINES, MAX_SCAN_LINES)
}

fn env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(value) => value.trim().parse::<usize>().unwrap_or(default),
        Err(_err) => default,
    }
}

/// Read hook input from stdin
pub fn read_input() -> Option<HookInput> {
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer).ok()?;
    if buffer.trim().is_empty() {
        return None;
    }
    serde_json::from_str(&buffer).ok()
}

/// Exit with pass (exit code 0, empty JSON)
pub fn pass() -> ! {
    println!("{{}}");
    process::exit(VerdictCode::from(Verdict::Pass).0 as i32)
}

/// Exit with warning (exit code 1, show message but allow)
pub fn warn(message: &str) -> ! {
    eprintln!("{message}");
    println!("{{}}");
    process::exit(VerdictCode::from(Verdict::Warn).0 as i32)
}

/// Exit with block (exit code 2, prevent action)
pub fn block(message: &str) -> ! {
    eprintln!("{message}");
    let output = HookOutput {
        decision: Some(Decision::from(Verdict::Block)),
        reason: Some(Reason(message.to_string())),
    };
    let _ = serde_json::to_writer(io::stdout(), &output);
    println!();
    process::exit(VerdictCode::from(Verdict::Block).0 as i32)
}

/// Require an edit-like tool; otherwise pass immediately.
pub fn require_edit_tool(tool_name: Option<ToolName>) {
    match tool_name {
        Some(ToolName::Edit) | Some(ToolName::WriteFile) | Some(ToolName::Replace) => {}
        Some(ToolName::ReadFile) => pass(),
        Some(ToolName::RunShellCommand) => pass(),
        Some(ToolName::ListDirectory) => pass(),
        Some(ToolName::Glob) => pass(),
        Some(ToolName::SearchFileContent) => pass(),
        Some(ToolName::Unknown(_name)) => pass(),
        None => pass(),
    }
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    for b in bytes {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Check if path is a Rust file
pub fn is_rust_file(path: &str) -> bool {
    path.ends_with(".rs")
}

/// Pass if path is not a Rust file
pub fn require_rust_file(path: &str) {
    if !is_rust_file(path) {
        pass();
    }
}

/// Check if path is a Python file
pub fn is_python_file(path: &str) -> bool {
    path.ends_with(".py")
}

/// Pass if path is not a Python file
pub fn require_python_file(path: &str) {
    if !is_python_file(path) {
        pass();
    }
}

/// Check if path is in a test context (tests/, benches/, or examples/)
pub fn is_test_path(path: &str) -> bool {
    path.contains("/tests/")
        || path.contains("/benches/")
        || path.contains("/examples/")
        || path.ends_with("_test.rs")
        || path.contains("/test_")
}

/// Check if content contains test context markers
pub fn has_test_context(content: &str) -> bool {
    content.contains("#[cfg(test)]") || content.contains("#[test]")
}

/// Check if a line is inside a test module or function
/// Simple heuristic: checks if #[test] or #[cfg(test)] appears before this line
pub fn line_in_test_context(content: &str, line_num: usize) -> bool {
    let lines: Vec<&str> = content.lines().collect();
    if line_num == 0 || line_num > lines.len() {
        return false;
    }

    // Look backwards for test markers
    let mut brace_depth = 0;
    for i in (0..line_num).rev() {
        let line = lines[i].trim();

        // Track brace depth (simplified)
        brace_depth += line.matches('}').count();
        brace_depth = brace_depth.saturating_sub(line.matches('{').count());

        // If we've exited all nested blocks, stop
        if brace_depth > 2 {
            break;
        }

        if line.contains("#[test]") || line.contains("#[cfg(test)]") {
            return true;
        }
    }
    false
}

/// Truncate a line to a maximum length, appending "..." when truncated.
pub fn truncate_line(line: &str, max_len: usize) -> String {
    let trimmed = line.trim();
    let count = trimmed.chars().count();
    if count > max_len {
        let take = max_len.saturating_sub(3);
        let truncated: String = trimmed.chars().take(take).collect();
        format!("{truncated}...")
    } else {
        trimmed.to_string()
    }
}

fn slice_to_char_boundary(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut last = 0;
    for (idx, _ch) in text.char_indices() {
        if idx > max_bytes {
            break;
        }
        last = idx;
    }
    &text[..last]
}

/// Find Cargo.toml by walking up from file path
pub fn find_cargo_toml(file_path: &str) -> Option<std::path::PathBuf> {
    let path = Path::new(file_path);
    let mut current = path.parent()?;

    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists() {
            return Some(cargo_toml);
        }
        match current.parent() {
            Some(parent) if parent != current => current = parent,
            Some(_parent) => return None,
            None => return None,
        }
    }
}

/// Extract crate name from Cargo.toml
pub fn get_crate_name(cargo_toml: &Path) -> Option<String> {
    let content = std::fs::read_to_string(cargo_toml).ok()?;
    let mut in_package = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "[package]" {
            in_package = true;
            continue;
        }
        if trimmed.starts_with('[') {
            in_package = false;
            continue;
        }
        if in_package
            && trimmed.starts_with("name")
            && let Some(name) = line.split('=').nth(1)
        {
            return Some(name.trim().trim_matches('"').trim_matches('\'').to_string());
        }
    }
    None
}

// ── Scan Primitives ───────────────────────────────────────────────

/// Compile a regex or exit with pass if invalid.
pub fn regex_or_pass(pattern: &str) -> Regex {
    match Regex::new(pattern) {
        Ok(re) => re,
        Err(_) => pass(),
    }
}

/// Result of scanning content lines.
///
/// # Tier: T2-C
/// Grounds to: T2-C(Violation)[] and T1(bool).
#[derive(Debug)]
pub struct ScanResult {
    pub violations: Vec<Violation>,
    pub truncated_lines: bool,
    pub truncated_hits: bool,
}

impl ScanResult {
    pub fn new() -> Self {
        Self {
            violations: Vec::new(),
            truncated_lines: false,
            truncated_hits: false,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.violations.is_empty()
    }

    fn push_violation(&mut self, evidence: EvidenceLine) -> bool {
        self.violations
            .push(Violation::new(evidence, Confidence::certain()));
        if self.violations.len() >= max_violations() {
            self.truncated_hits = true;
            return true;
        }
        false
    }
}

impl Default for ScanResult {
    fn default() -> Self {
        Self::new()
    }
}

fn should_skip_line(
    context_filter: Option<fn(&str, usize) -> bool>,
    text: &str,
    line_num: usize,
) -> bool {
    match context_filter {
        Some(filter) => filter(text, line_num),
        None => false,
    }
}

/// Scan content lines against a single regex.
///
/// `kind` labels each violation. `context_filter` optionally skips lines
/// where `filter(full_text, line_num) == true` (e.g. test context).
pub fn scan_lines(
    content: &ContentSlice<'_>,
    regex: &Regex,
    kind: &str,
    context_filter: Option<fn(&str, usize) -> bool>,
) -> ScanResult {
    let mut result = ScanResult::new();
    let max_lines = max_scan_lines();
    let snip_len = snippet_len();

    for (line_num, line) in content.text.lines().enumerate() {
        if line_num >= max_lines {
            result.truncated_lines = true;
            break;
        }
        if !regex.is_match(line) {
            continue;
        }
        if should_skip_line(context_filter, content.text, line_num) {
            continue;
        }
        let snippet = truncate_line(line, snip_len);
        let evidence = EvidenceLine::new(line_num + 1, kind, snippet);
        if result.push_violation(evidence) {
            break;
        }
    }
    result
}

fn is_comment_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with("//") || trimmed.starts_with("///")
}

/// Scan content lines against multiple regex patterns.
///
/// Each entry in `patterns` is `(kind_label, regex)`.
/// If `skip_comments` is true, lines starting with `//` are skipped.
pub fn scan_lines_multi(
    content: &ContentSlice<'_>,
    patterns: &[(&str, &Regex)],
    skip_comments: bool,
) -> ScanResult {
    let mut result = ScanResult::new();
    let max_lines = max_scan_lines();
    let snip_len = snippet_len();

    for (line_num, line) in content.text.lines().enumerate() {
        if line_num >= max_lines {
            result.truncated_lines = true;
            break;
        }
        if skip_comments && is_comment_line(line) {
            continue;
        }
        let hit = scan_line_patterns(line, line_num, patterns, snip_len, &mut result);
        if hit {
            break;
        }
    }
    result
}

fn scan_line_patterns(
    line: &str,
    line_num: usize,
    patterns: &[(&str, &Regex)],
    snip_len: usize,
    result: &mut ScanResult,
) -> bool {
    for &(kind, regex) in patterns {
        if !regex.is_match(line) {
            continue;
        }
        let snippet = truncate_line(line, snip_len);
        let evidence = EvidenceLine::new(line_num + 1, kind, snippet);
        if result.push_violation(evidence) {
            return true;
        }
    }
    false
}

/// Shared secret detection patterns. Single source of truth used by
/// both `secret-scanner` and `pretool-dispatcher`.
pub fn secret_patterns() -> Vec<(&'static str, Regex)> {
    let specs: &[(&str, &str)] = &[
        ("AWS Access Key", r"AKIA[0-9A-Z]{16}"),
        ("GitHub PAT", r"ghp_[A-Za-z0-9_]{36}"),
        ("GitHub OAuth", r"gho_[A-Za-z0-9_]{36}"),
        ("OpenAI API Key", r"sk-[A-Za-z0-9]{40,}"),
        ("Stripe Secret Key", r"sk_live_[A-Za-z0-9]{24,}"),
        (
            "Private Key",
            r"-----BEGIN (RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY",
        ),
        ("Database URL", r"(postgres|mysql|mongodb)://[^:]+:[^@]+@"),
        (
            "Slack Webhook",
            r"https://hooks\.slack\.com/services/[A-Z0-9/]+",
        ),
        (
            "Generic API Key",
            r#"(?i)(api[_-]?key|apikey)\s*[:=]\s*["'][A-Za-z0-9]{20,}["']"#,
        ),
        (
            "Generic Secret",
            r#"(?i)(secret|password|token)\s*[:=]\s*["'][A-Za-z0-9!@#$%^&*]{16,}["']"#,
        ),
    ];

    specs
        .iter()
        .filter_map(|(name, pat)| Regex::new(pat).ok().map(|re| (*name, re)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_rust_file() {
        assert!(is_rust_file("main.rs"));
        assert!(is_rust_file("/path/to/lib.rs"));
        assert!(!is_rust_file("main.py"));
        assert!(!is_rust_file("Cargo.toml"));
    }

    #[test]
    fn test_is_python_file() {
        assert!(is_python_file("script.py"));
        assert!(!is_python_file("main.rs"));
    }

    #[test]
    fn test_regex_or_pass_valid() {
        let re = regex_or_pass(r"hello\s+world");
        assert!(re.is_match("hello  world"));
    }

    #[test]
    fn test_scan_lines_finds_match() {
        let content = ContentSlice {
            text: "line one\npanic!(\"boom\")\nline three",
            total_bytes: 35,
            scanned_bytes: 35,
            truncated: false,
        };
        let re = regex_or_pass(r"panic!\s*\(");
        let result = scan_lines(&content, &re, "panic!", None);
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.violations[0].evidence.line.0, 2);
    }

    #[test]
    fn test_scan_lines_skips_with_filter() {
        let content = ContentSlice {
            text: "fn test_it() {\npanic!(\"ok\")\n}",
            total_bytes: 30,
            scanned_bytes: 30,
            truncated: false,
        };
        let re = regex_or_pass(r"panic!\s*\(");
        let result = scan_lines(&content, &re, "panic!", Some(|_text, _line| true));
        assert!(result.is_empty());
    }

    #[test]
    fn test_scan_lines_multi_finds_all() {
        let content = ContentSlice {
            text: "Command::new(\"bash\")\n.arg(format!(\"x\"))\nclean line",
            total_bytes: 50,
            scanned_bytes: 50,
            truncated: false,
        };
        let re1 = regex_or_pass(r#"Command::new\s*\(\s*"bash""#);
        let re2 = regex_or_pass(r#"\.arg\s*\(\s*format!"#);
        let patterns: Vec<(&str, &Regex)> = vec![("shell", &re1), ("dynamic arg", &re2)];
        let result = scan_lines_multi(&content, &patterns, false);
        assert_eq!(result.violations.len(), 2);
    }

    #[test]
    fn test_scan_lines_multi_skips_comments() {
        let content = ContentSlice {
            text: "// Command::new(\"bash\")\nreal code",
            total_bytes: 35,
            scanned_bytes: 35,
            truncated: false,
        };
        let re = regex_or_pass(r#"Command::new"#);
        let patterns: Vec<(&str, &Regex)> = vec![("shell", &re)];
        let result = scan_lines_multi(&content, &patterns, true);
        assert!(result.is_empty());
    }

    #[test]
    fn test_secret_patterns_all_compile() {
        let patterns = secret_patterns();
        assert_eq!(patterns.len(), 10);
        for (name, _re) in &patterns {
            assert!(!name.is_empty());
        }
    }

    #[test]
    fn test_scan_result_empty() {
        let result = ScanResult::new();
        assert!(result.is_empty());
        assert!(!result.truncated_lines);
        assert!(!result.truncated_hits);
    }
}
