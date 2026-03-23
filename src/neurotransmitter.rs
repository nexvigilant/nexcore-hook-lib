//! Neurotransmitter Pattern for Claude Code Hooks
//!
//! # Design Principle
//!
//! Hooks should be **sub-millisecond**. Heavy work happens elsewhere.
//!
//! ```text
//! Traditional Hook:     Hook --[work]--[work]--[work]--> return (slow)
//! Neurotransmitter:     Hook --[emit signal]--> return (fast)
//!                              |
//!                              +---> Receiver processes async
//! ```
//!
//! # T1 Primitive Grounding
//!
//! | Concept | T1 Primitive | Symbol |
//! |---------|--------------|--------|
//! | Signal | Existence | there exists |
//! | Emission | Causality | -> |
//! | Timestamp | Sequence | sigma |
//! | Fire-and-forget | Irreversibility | alpha |
//! | Priority | Ordering | < |
//! | Vesicle | Aggregation | Union |
//! | Circuit | State | varsigma |
//!
//! # Biological Neurotransmitter Analogy
//!
//! ```text
//! Biology:     Vesicle --[dock]--> Synaptic cleft --[receptor]--> Response
//! This System: Vesicle --[flush]--> Signal file   --[receiver]--> Metrics
//! ```
//!
//! ## Vesicle Batching
//!
//! Like biological vesicles that accumulate neurotransmitters before release,
//! this system batches signals in memory before flushing to disk. Benefits:
//! - Amortizes file I/O overhead across multiple signals
//! - Single syscall for N signals instead of N syscalls
//! - Memory-only until flush (sub-microsecond per signal)
//!
//! ## Circuit Breaker
//!
//! Like synaptic fatigue that prevents overstimulation, the circuit breaker
//! stops emission when the receiver appears dead. This prevents:
//! - Pointless I/O to a non-functional receiver
//! - Resource exhaustion from unbounded file growth
//! - Latency spikes from failing writes
//!
//! # Usage
//!
//! ```rust,ignore
//! use nexcore_hook_lib::neurotransmitter::{Signal, emit, Priority};
//!
//! // Standard emission - batched automatically
//! emit(Signal::new("skill_invoked")
//!     .with_data("skill", "primitive-extractor"));
//!
//! // Critical signal - bypasses batching, writes immediately
//! emit(Signal::new("tool_blocked")
//!     .with_priority(Priority::Critical)
//!     .with_data("tool", "Write"));
//!
//! // At hook end - flush any buffered signals
//! flush();
//! ```

use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Signal channel path
const SIGNAL_PATH: &str = "/home/matthew/.claude/brain/telemetry/signals.jsonl";

/// Maximum signal size in bytes (prevents runaway data)
const MAX_SIGNAL_BYTES: usize = 4096;

/// Maximum signals in a vesicle before forced flush
const VESICLE_CAPACITY: usize = 16;

/// Maximum bytes in a vesicle before forced flush
const VESICLE_MAX_BYTES: usize = 8192;

/// Circuit breaker: consecutive failures before opening
const CIRCUIT_FAILURE_THRESHOLD: u8 = 3;

/// Circuit breaker: time to wait before attempting recovery
const CIRCUIT_RECOVERY_MS: u64 = 5000;

// ============================================================================
// T2-P Primitives
// ============================================================================

/// Signal priority level.
///
/// # Tier: T2-P
/// Grounds to: T1(u8) via repr.
/// Ord: Telemetry < Normal < High < Critical.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    Telemetry = 0,
    #[default]
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Circuit breaker state.
///
/// # Tier: T2-P
/// Grounds to: T1(u8) via repr.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CircuitState {
    #[default]
    Closed = 0,
    HalfOpen = 1,
    Open = 2,
}

// ============================================================================
// T2-C Composites
// ============================================================================

/// A lightweight signal emitted by hooks.
///
/// # Tier: T2-C
/// Grounds to: T1(String, u128) via signal_type and timestamp,
/// plus T1(String) via data HashMap, plus T2-P(Priority).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub signal_type: String,
    pub timestamp_ms: u128,
    #[serde(default, skip_serializing_if = "is_default_priority")]
    pub priority: Priority,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub data: HashMap<String, String>,
}

fn is_default_priority(p: &Priority) -> bool {
    *p == Priority::Normal
}

impl Signal {
    #[inline]
    pub fn new(signal_type: impl Into<String>) -> Self {
        Self {
            signal_type: signal_type.into(),
            timestamp_ms: current_timestamp_ms(),
            priority: Priority::default(),
            data: HashMap::new(),
        }
    }

    #[must_use]
    #[inline]
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    #[must_use]
    #[inline]
    pub fn with_data(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.data.insert(key.into(), value.into());
        self
    }

    fn to_bytes(&self) -> Option<Vec<u8>> {
        let json = serde_json::to_string(self).ok()?;
        if json.len() > MAX_SIGNAL_BYTES {
            return None;
        }
        let mut bytes = json.into_bytes();
        bytes.push(b'\n');
        Some(bytes)
    }
}

fn current_timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// Vesicle: batched signal container.
///
/// # Tier: T2-C
/// Grounds to: T2-C(Signal)[] via Vec, plus T1(usize) via byte tracking.
#[derive(Debug, Default)]
struct Vesicle {
    buffer: Vec<u8>,
    count: usize,
}

impl Vesicle {
    fn add(&mut self, bytes: Vec<u8>) -> bool {
        self.buffer.extend(bytes);
        self.count += 1;
        self.should_flush()
    }

    fn should_flush(&self) -> bool {
        self.count >= VESICLE_CAPACITY || self.buffer.len() >= VESICLE_MAX_BYTES
    }

    fn is_empty(&self) -> bool {
        self.count == 0
    }

    fn take(&mut self) -> Vec<u8> {
        self.count = 0;
        std::mem::take(&mut self.buffer)
    }
}

/// Circuit breaker for emission control.
///
/// # Tier: T2-C
/// Grounds to: T2-P(CircuitState) plus T1(u8, Instant).
#[derive(Debug, Default)]
struct CircuitBreaker {
    state: CircuitState,
    consecutive_failures: u8,
    last_failure: Option<Instant>,
}

impl CircuitBreaker {
    fn should_emit(&mut self, priority: Priority) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => self.check_open_state(priority),
            CircuitState::HalfOpen => true,
        }
    }

    fn check_open_state(&mut self, priority: Priority) -> bool {
        if priority == Priority::Critical {
            return true;
        }
        self.try_recovery()
    }

    fn try_recovery(&mut self) -> bool {
        let Some(last) = self.last_failure else {
            return false;
        };
        if last.elapsed() >= Duration::from_millis(CIRCUIT_RECOVERY_MS) {
            self.state = CircuitState::HalfOpen;
            return true;
        }
        false
    }

    fn record_success(&mut self) {
        self.consecutive_failures = 0;
        self.state = CircuitState::Closed;
    }

    fn record_failure(&mut self) {
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        self.last_failure = Some(Instant::now());
        if self.consecutive_failures >= CIRCUIT_FAILURE_THRESHOLD {
            self.state = CircuitState::Open;
        }
    }
}

/// Thread-local emitter state.
#[derive(Debug, Default)]
struct EmitterState {
    vesicle: Vesicle,
    circuit: CircuitBreaker,
}

thread_local! {
    static EMITTER: RefCell<EmitterState> = RefCell::new(EmitterState::default());
}

// ============================================================================
// Emission API
// ============================================================================

/// Emit a signal to the telemetry channel.
///
/// **Batched fire-and-forget**: Signals are accumulated in a vesicle and
/// flushed together when the vesicle is full or on explicit flush().
pub fn emit(signal: Signal) {
    let priority = signal.priority;
    let Some(bytes) = signal.to_bytes() else {
        return;
    };
    emit_with_priority(bytes, priority);
}

fn emit_with_priority(bytes: Vec<u8>, priority: Priority) {
    EMITTER.with(|cell| {
        let mut state = cell.borrow_mut();
        if !state.circuit.should_emit(priority) {
            return;
        }
        if priority == Priority::Critical {
            emit_critical(&mut state, bytes);
        } else {
            emit_batched(&mut state, bytes);
        }
    });
}

fn emit_critical(state: &mut EmitterState, bytes: Vec<u8>) {
    flush_vesicle(state);
    emit_buffer(&bytes, &mut state.circuit);
}

fn emit_batched(state: &mut EmitterState, bytes: Vec<u8>) {
    if state.vesicle.add(bytes) {
        flush_vesicle(state);
    }
}

fn flush_vesicle(state: &mut EmitterState) {
    if state.vesicle.is_empty() {
        return;
    }
    let buffer = state.vesicle.take();
    emit_buffer(&buffer, &mut state.circuit);
}

/// Flush any buffered signals.
pub fn flush() {
    EMITTER.with(|cell| {
        let mut state = cell.borrow_mut();
        flush_vesicle(&mut state);
    });
}

fn emit_buffer(buffer: &[u8], circuit: &mut CircuitBreaker) {
    match emit_bytes(buffer) {
        Ok(()) => circuit.record_success(),
        Err(_) => circuit.record_failure(),
    }
}

fn emit_bytes(bytes: &[u8]) -> std::io::Result<()> {
    let path = signal_path();
    ensure_parent_dir(&path)?;
    let mut file = open_signal_file(&path)?;
    file.write_all(bytes)
}

fn ensure_parent_dir(path: &std::path::Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn open_signal_file(path: &std::path::Path) -> std::io::Result<File> {
    OpenOptions::new().create(true).append(true).open(path)
}

fn signal_path() -> PathBuf {
    std::env::var("CLAUDE_SIGNAL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(SIGNAL_PATH))
}

/// Get current circuit breaker state (for diagnostics).
pub fn circuit_state() -> CircuitState {
    EMITTER.with(|cell| cell.borrow().circuit.state)
}

/// Get current vesicle count (for diagnostics).
pub fn vesicle_count() -> usize {
    EMITTER.with(|cell| cell.borrow().vesicle.count)
}

// ============================================================================
// Predefined Signal Types
// ============================================================================

pub mod signals {
    pub const SKILL_INVOKED: &str = "skill_invoked";
    pub const TOOL_BLOCKED: &str = "tool_blocked";
    pub const TOOL_COMPLETED: &str = "tool_completed";
    pub const HOOK_START: &str = "hook_start";
    pub const HOOK_END: &str = "hook_end";
    pub const SESSION_START: &str = "session_start";
    pub const SESSION_END: &str = "session_end";
    pub const ERROR: &str = "error";
    pub const METRIC: &str = "metric";
    pub const CIRCUIT_STATE: &str = "circuit_state";
}

// ============================================================================
// Convenience Emitters
// ============================================================================

#[inline]
pub fn emit_skill_invoked(skill: &str, session_id: Option<&str>) {
    let mut signal = Signal::new(signals::SKILL_INVOKED).with_data("skill", skill);
    if let Some(sid) = session_id {
        signal = signal.with_data("session", sid);
    }
    emit(signal);
}

#[inline]
pub fn emit_tool_blocked(tool: &str, hook: &str, reason: &str) {
    emit(
        Signal::new(signals::TOOL_BLOCKED)
            .with_priority(Priority::High)
            .with_data("tool", tool)
            .with_data("hook", hook)
            .with_data("reason", reason),
    );
}

#[inline]
pub fn emit_metric(name: &str, value: f64) {
    emit(
        Signal::new(signals::METRIC)
            .with_priority(Priority::Telemetry)
            .with_data("name", name)
            .with_data("value", value.to_string()),
    );
}

#[inline]
pub fn emit_hook_timing(hook: &str, duration_us: u64) {
    emit(
        Signal::new(signals::HOOK_END)
            .with_data("hook", hook)
            .with_data("duration_us", duration_us.to_string()),
    );
    flush();
}

#[inline]
pub fn emit_error(hook: &str, error: &str) {
    emit(
        Signal::new(signals::ERROR)
            .with_priority(Priority::High)
            .with_data("hook", hook)
            .with_data("error", error),
    );
}

// ============================================================================
// Signal Reader (for receivers/analyzers)
// ============================================================================

pub fn read_signals(path: Option<&str>) -> std::io::Result<Vec<Signal>> {
    let path = path.map(PathBuf::from).unwrap_or_else(signal_path);
    let content = std::fs::read_to_string(path)?;
    Ok(content
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect())
}

pub fn rotate_signals() -> std::io::Result<()> {
    let path = signal_path();
    if !path.exists() {
        return Ok(());
    }
    let timestamp = current_timestamp_ms() / 1000;
    let archive = path.with_extension(format!("jsonl.{timestamp}"));
    std::fs::rename(&path, archive)?;
    File::create(path)?;
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_signal_creation_is_fast() {
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = Signal::new("test")
                .with_priority(Priority::Normal)
                .with_data("key", "value")
                .with_data("another", "data");
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 10,
            "Signal creation too slow: {:?}",
            elapsed
        );
    }

    #[test]
    fn test_signal_serialization() {
        let signal = Signal::new("test_event")
            .with_data("skill", "extractor")
            .with_data("session", "abc123");
        let bytes = signal.to_bytes().expect("should serialize");
        assert!(bytes.len() < MAX_SIGNAL_BYTES);
        assert!(bytes.ends_with(b"\n"));
    }

    #[test]
    fn test_signal_with_priority_serialization() {
        let signal = Signal::new("critical_event").with_priority(Priority::Critical);
        let bytes = signal.to_bytes().expect("should serialize");
        let json = String::from_utf8_lossy(&bytes);
        assert!(json.contains("critical"));
    }

    #[test]
    fn test_default_priority_not_serialized() {
        let signal = Signal::new("normal_event");
        let bytes = signal.to_bytes().expect("should serialize");
        let json = String::from_utf8_lossy(&bytes);
        assert!(!json.contains("priority"));
    }

    #[test]
    fn test_large_signal_dropped() {
        let mut signal = Signal::new("test");
        for i in 0..100 {
            signal = signal.with_data(format!("key{i}"), "x".repeat(100));
        }
        assert!(signal.to_bytes().is_none());
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Telemetry < Priority::Normal);
        assert!(Priority::Normal < Priority::High);
        assert!(Priority::High < Priority::Critical);
    }

    #[test]
    fn test_vesicle_capacity() {
        let mut vesicle = Vesicle::default();
        let small_signal = b"{\"type\":\"test\"}\n".to_vec();
        for i in 0..VESICLE_CAPACITY {
            let should_flush = vesicle.add(small_signal.clone());
            if i < VESICLE_CAPACITY - 1 {
                assert!(!should_flush, "Should not flush before capacity");
            } else {
                assert!(should_flush, "Should flush at capacity");
            }
        }
    }

    #[test]
    fn test_vesicle_byte_limit() {
        let mut vesicle = Vesicle::default();
        let large_signal = format!("{{\"data\":\"{}\"}}\n", "x".repeat(VESICLE_MAX_BYTES / 2));
        vesicle.add(large_signal.as_bytes().to_vec());
        let should_flush = vesicle.add(large_signal.as_bytes().to_vec());
        assert!(should_flush, "Should flush when byte limit exceeded");
    }

    #[test]
    fn test_circuit_breaker_opens_after_failures() {
        let mut circuit = CircuitBreaker::default();
        for _ in 0..CIRCUIT_FAILURE_THRESHOLD {
            circuit.record_failure();
        }
        assert_eq!(circuit.state, CircuitState::Open);
        assert!(!circuit.should_emit(Priority::Normal));
        assert!(circuit.should_emit(Priority::Critical));
    }

    #[test]
    fn test_circuit_breaker_closes_on_success() {
        let mut circuit = CircuitBreaker::default();
        for _ in 0..CIRCUIT_FAILURE_THRESHOLD {
            circuit.record_failure();
        }
        assert_eq!(circuit.state, CircuitState::Open);
        circuit.state = CircuitState::HalfOpen;
        circuit.record_success();
        assert_eq!(circuit.state, CircuitState::Closed);
        assert!(circuit.should_emit(Priority::Normal));
    }

    #[test]
    fn test_convenience_emitters_dont_panic() {
        emit_skill_invoked("test-skill", Some("session-123"));
        emit_tool_blocked("Write", "blocker", "reason");
        emit_metric("latency_ms", 42.5);
        emit_error("test-hook", "test error");
        flush();
    }

    #[test]
    fn test_emit_and_flush() {
        for i in 0..5 {
            emit(Signal::new("test").with_data("index", i.to_string()));
        }
        flush();
        assert_eq!(vesicle_count(), 0);
    }

    #[test]
    fn test_critical_signal_immediate_flush() {
        emit(Signal::new("normal1"));
        emit(Signal::new("normal2"));
        emit(Signal::new("critical").with_priority(Priority::Critical));
        assert_eq!(vesicle_count(), 0);
    }
}
