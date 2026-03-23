//! Cytokine Bridge for Claude Code Hooks
//!
//! Enables hooks to emit cytokine signals to the Guardian homeostasis loop.
//!
//! # T1 Primitive Grounding
//!
//! | Concept | T1 Primitive | Symbol |
//! |---------|--------------|--------|
//! | Bridge | Mapping | mu |
//! | Emission | Causality | -> |
//! | Family | Sum type | Sigma |
//! | Severity | Quantity | N |
//!
//! # Integration Pattern
//!
//! ```text
//! Hook Event        Cytokine Family       Guardian Response
//! ─────────────────────────────────────────────────────────
//! tool_blocked   -> TNF-alpha (terminate) -> Block actuator
//! check_failed   -> IL-6 (acute)          -> Alert actuator
//! skill_invoked  -> IL-2 (growth)         -> Bond strengthening
//! error_detected -> IL-1 (alarm)          -> Escalation
//! rate_exceeded  -> IL-10 (suppress)      -> Rate limit
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use nexcore_hook_lib::cytokine::{emit_hook_cytokine, HookEvent};
//!
//! // Emit when a tool is blocked
//! emit_hook_cytokine(HookEvent::ToolBlocked {
//!     tool: "Write".to_string(),
//!     hook: "unwrap-guardian".to_string(),
//!     reason: "unwrap() detected".to_string(),
//! });
//! ```

use crate::neurotransmitter::{self, Priority, Signal};
use nexcore_cytokine::{Cytokine, CytokineFamily, Emitter, Scope, ThreatLevel, global_bus};

/// Hook events that can be emitted as cytokines.
///
/// # Tier: T2-P (Sum Type)
/// Each variant maps to a specific cytokine family.
#[derive(Debug, Clone)]
pub enum HookEvent {
    /// Tool was blocked by a hook -> TNF-alpha (terminate)
    ToolBlocked {
        tool: String,
        hook: String,
        reason: String,
    },
    /// Compilation/check failed -> IL-6 (acute response)
    CheckFailed { hook: String, error: String },
    /// Skill was invoked -> IL-2 (growth/proliferation)
    SkillInvoked {
        skill: String,
        session_id: Option<String>,
    },
    /// Error detected -> IL-1 (alarm)
    ErrorDetected {
        hook: String,
        error: String,
        severity: HookSeverity,
    },
    /// Rate/threshold exceeded -> IL-10 (suppress)
    ThresholdExceeded {
        metric: String,
        value: f64,
        threshold: f64,
    },
    /// Hook completed successfully -> TGF-beta (regulation)
    HookCompleted {
        hook: String,
        duration_us: u64,
        verdict: String,
    },
    /// Custom cytokine emission
    Custom {
        family: String,
        name: String,
        severity: HookSeverity,
        payload: Option<serde_json::Value>,
    },
}

/// Hook severity levels mapped to cytokine severity.
///
/// # Tier: T2-P
/// Grounds to T1(u8) via match.
#[derive(Debug, Clone, Copy, Default)]
pub enum HookSeverity {
    Trace,
    Low,
    #[default]
    Medium,
    High,
    Critical,
}

impl From<HookSeverity> for ThreatLevel {
    fn from(s: HookSeverity) -> Self {
        match s {
            HookSeverity::Trace => ThreatLevel::Trace,
            HookSeverity::Low => ThreatLevel::Low,
            HookSeverity::Medium => ThreatLevel::Medium,
            HookSeverity::High => ThreatLevel::High,
            HookSeverity::Critical => ThreatLevel::Critical,
        }
    }
}

/// Map cytokine severity to neurotransmitter priority.
///
/// # Tier: T2-P (Mapping)
/// Grounds to: T1(u8) via repr on both sides.
fn severity_to_priority(severity: &ThreatLevel) -> Priority {
    match severity {
        ThreatLevel::Critical => Priority::Critical,
        ThreatLevel::High => Priority::High,
        ThreatLevel::Medium | ThreatLevel::Low => Priority::Normal,
        ThreatLevel::Trace => Priority::Telemetry,
    }
}

/// Convert a family enum to a snake_case identifier for signal_type.
///
/// Uses lowercase, filesystem-safe names (no Greek letters).
fn family_to_slug(family: &CytokineFamily) -> String {
    match family {
        CytokineFamily::Il1 => "il1".to_string(),
        CytokineFamily::Il2 => "il2".to_string(),
        CytokineFamily::Il6 => "il6".to_string(),
        CytokineFamily::Il10 => "il10".to_string(),
        CytokineFamily::TnfAlpha => "tnf_alpha".to_string(),
        CytokineFamily::IfnGamma => "ifn_gamma".to_string(),
        CytokineFamily::TgfBeta => "tgf_beta".to_string(),
        CytokineFamily::Csf => "csf".to_string(),
        CytokineFamily::Custom(id) => format!("custom_{id}"),
    }
}

/// Convert a Cytokine to a neurotransmitter Signal for file persistence.
///
/// # Signal Type Format
/// `cytokine:{family_slug}:{name}`
///
/// # Tier: T2-C (Composite Mapping)
/// Cytokine → Signal is a lossy projection that preserves:
/// - family (via signal_type prefix)
/// - severity → priority mapping
/// - payload flattened to data HashMap<String, String>
/// - source, scope, id as metadata
pub fn cytokine_to_signal(c: &Cytokine) -> Signal {
    let family_slug = family_to_slug(&c.family);
    let signal_type = format!("cytokine:{family_slug}:{}", c.name);
    let priority = severity_to_priority(&c.severity);

    let mut signal = Signal::new(signal_type).with_priority(priority);

    // Core metadata
    signal = signal
        .with_data("cytokine_id", &c.id)
        .with_data("family", &family_slug)
        .with_data("severity", c.severity.to_string())
        .with_data("scope", c.scope.to_string());

    if let Some(src) = &c.source {
        signal = signal.with_data("source", src.as_str());
    }

    // Flatten payload object keys into data map
    if let serde_json::Value::Object(map) = &c.payload {
        for (k, v) in map {
            let val_str = match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            signal = signal.with_data(format!("payload_{k}"), val_str);
        }
    }

    signal
}

/// Emit a hook event as a cytokine signal.
///
/// **Dual-emit**: Writes to `signals.jsonl` via neurotransmitter (file-based,
/// persistent) AND to the in-memory cytokine bus (for same-process consumers).
///
/// The neurotransmitter write happens first (synchronous, thread-local vesicle).
/// The bus emit happens second (async, may silently fail in short-lived processes).
///
/// # T1 Grounding
/// - → (causality): HookEvent causes cytokine emission
/// - μ (mapping): Event type determines cytokine family
/// - ∂ (fork): Dual-path emission (file + bus)
pub fn emit_hook_cytokine(event: HookEvent) {
    let cytokine = event_to_cytokine(event);

    // Path 1: Persistent file write via neurotransmitter
    // This is the primary path — signals.jsonl is consumed by signal-receiver daemon
    let signal = cytokine_to_signal(&cytokine);
    neurotransmitter::emit(signal);
    neurotransmitter::flush();

    // Path 2: In-memory bus (for same-process consumers, usually fires into void)
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        let _ = handle.block_on(async {
            let bus = global_bus();
            bus.emit(cytokine).await
        });
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build();
        if let Ok(rt) = rt {
            let _ = rt.block_on(async {
                let bus = global_bus();
                bus.emit(cytokine).await
            });
        }
    }
}

/// Convert a hook event to a cytokine signal.
fn event_to_cytokine(event: HookEvent) -> Cytokine {
    match event {
        HookEvent::ToolBlocked { tool, hook, reason } => {
            Cytokine::new(CytokineFamily::TnfAlpha, format!("hook_blocked:{}", hook))
                .with_severity(ThreatLevel::High)
                .with_scope(Scope::Endocrine)
                .with_payload(serde_json::json!({
                    "event": "tool_blocked",
                    "tool": tool,
                    "hook": hook,
                    "reason": reason,
                }))
                .with_source("hook-lib")
        }
        HookEvent::CheckFailed { hook, error } => {
            Cytokine::new(CytokineFamily::Il6, format!("hook_check_failed:{}", hook))
                .with_severity(ThreatLevel::High)
                .with_scope(Scope::Endocrine)
                .with_payload(serde_json::json!({
                    "event": "check_failed",
                    "hook": hook,
                    "error": error,
                }))
                .with_source("hook-lib")
        }
        HookEvent::SkillInvoked { skill, session_id } => {
            Cytokine::new(CytokineFamily::Il2, format!("skill_invoked:{}", skill))
                .with_severity(ThreatLevel::Low)
                .with_scope(Scope::Paracrine)
                .with_payload(serde_json::json!({
                    "event": "skill_invoked",
                    "skill": skill,
                    "session_id": session_id,
                }))
                .with_source("hook-lib")
        }
        HookEvent::ErrorDetected {
            hook,
            error,
            severity,
        } => Cytokine::new(CytokineFamily::Il1, format!("hook_error:{}", hook))
            .with_severity(severity.into())
            .with_scope(Scope::Systemic)
            .with_payload(serde_json::json!({
                "event": "error_detected",
                "hook": hook,
                "error": error,
            }))
            .with_source("hook-lib"),
        HookEvent::ThresholdExceeded {
            metric,
            value,
            threshold,
        } => Cytokine::new(
            CytokineFamily::Il10,
            format!("threshold_exceeded:{}", metric),
        )
        .with_severity(ThreatLevel::Medium)
        .with_scope(Scope::Paracrine)
        .with_payload(serde_json::json!({
            "event": "threshold_exceeded",
            "metric": metric,
            "value": value,
            "threshold": threshold,
        }))
        .with_source("hook-lib"),
        HookEvent::HookCompleted {
            hook,
            duration_us,
            verdict,
        } => Cytokine::new(CytokineFamily::TgfBeta, format!("hook_completed:{}", hook))
            .with_severity(ThreatLevel::Trace)
            .with_scope(Scope::Autocrine)
            .with_payload(serde_json::json!({
                "event": "hook_completed",
                "hook": hook,
                "duration_us": duration_us,
                "verdict": verdict,
            }))
            .with_source("hook-lib"),
        HookEvent::Custom {
            family,
            name,
            severity,
            payload,
        } => {
            let cytokine_family = parse_family(&family);
            let mut cytokine = Cytokine::new(cytokine_family, name)
                .with_severity(severity.into())
                .with_scope(Scope::Paracrine)
                .with_source("hook-lib");
            if let Some(p) = payload {
                cytokine = cytokine.with_payload(p);
            }
            cytokine
        }
    }
}

/// Parse a family string to CytokineFamily.
fn parse_family(s: &str) -> CytokineFamily {
    match s.to_lowercase().as_str() {
        "il1" | "il-1" => CytokineFamily::Il1,
        "il2" | "il-2" => CytokineFamily::Il2,
        "il6" | "il-6" => CytokineFamily::Il6,
        "il10" | "il-10" => CytokineFamily::Il10,
        "tnf_alpha" | "tnf-alpha" | "tnf" => CytokineFamily::TnfAlpha,
        "ifn_gamma" | "ifn-gamma" | "ifn" => CytokineFamily::IfnGamma,
        "tgf_beta" | "tgf-beta" | "tgf" => CytokineFamily::TgfBeta,
        "csf" => CytokineFamily::Csf,
        // Custom families use hash of name as u16 identifier
        other => CytokineFamily::Custom(fnv1a_16(other.as_bytes())),
    }
}

/// FNV-1a hash to 16-bit for custom cytokine family identifiers.
fn fnv1a_16(bytes: &[u8]) -> u16 {
    const FNV_OFFSET: u32 = 0x811c9dc5;
    const FNV_PRIME: u32 = 0x01000193;
    let mut hash = FNV_OFFSET;
    for b in bytes {
        hash ^= *b as u32;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    // Fold 32-bit to 16-bit via XOR
    ((hash >> 16) ^ (hash & 0xFFFF)) as u16
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Emit a tool blocked cytokine (TNF-alpha).
#[inline]
pub fn emit_tool_blocked(tool: &str, hook: &str, reason: &str) {
    emit_hook_cytokine(HookEvent::ToolBlocked {
        tool: tool.to_string(),
        hook: hook.to_string(),
        reason: reason.to_string(),
    });
}

/// Emit a check failed cytokine (IL-6).
#[inline]
pub fn emit_check_failed(hook: &str, error: &str) {
    emit_hook_cytokine(HookEvent::CheckFailed {
        hook: hook.to_string(),
        error: error.to_string(),
    });
}

/// Emit a skill invoked cytokine (IL-2).
#[inline]
pub fn emit_skill_invoked(skill: &str, session_id: Option<&str>) {
    emit_hook_cytokine(HookEvent::SkillInvoked {
        skill: skill.to_string(),
        session_id: session_id.map(String::from),
    });
}

/// Emit an error detected cytokine (IL-1).
#[inline]
pub fn emit_error(hook: &str, error: &str, severity: HookSeverity) {
    emit_hook_cytokine(HookEvent::ErrorDetected {
        hook: hook.to_string(),
        error: error.to_string(),
        severity,
    });
}

/// Emit a threshold exceeded cytokine (IL-10).
#[inline]
pub fn emit_threshold_exceeded(metric: &str, value: f64, threshold: f64) {
    emit_hook_cytokine(HookEvent::ThresholdExceeded {
        metric: metric.to_string(),
        value,
        threshold,
    });
}

/// Emit a hook completed cytokine (TGF-beta).
#[inline]
pub fn emit_hook_completed(hook: &str, duration_us: u64, verdict: &str) {
    emit_hook_cytokine(HookEvent::HookCompleted {
        hook: hook.to_string(),
        duration_us,
        verdict: verdict.to_string(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hook_severity_conversion() {
        assert!(matches!(
            ThreatLevel::from(HookSeverity::Trace),
            ThreatLevel::Trace
        ));
        assert!(matches!(
            ThreatLevel::from(HookSeverity::Critical),
            ThreatLevel::Critical
        ));
    }

    #[test]
    fn test_parse_family() {
        assert!(matches!(parse_family("il1"), CytokineFamily::Il1));
        assert!(matches!(
            parse_family("TNF-alpha"),
            CytokineFamily::TnfAlpha
        ));
        assert!(matches!(parse_family("unknown"), CytokineFamily::Custom(_)));
    }

    #[test]
    fn test_event_to_cytokine_tool_blocked() {
        let event = HookEvent::ToolBlocked {
            tool: "Write".to_string(),
            hook: "test-hook".to_string(),
            reason: "test reason".to_string(),
        };
        let cytokine = event_to_cytokine(event);
        assert!(matches!(cytokine.family, CytokineFamily::TnfAlpha));
        assert!(matches!(cytokine.severity, ThreatLevel::High));
    }

    #[test]
    fn test_event_to_cytokine_skill_invoked() {
        let event = HookEvent::SkillInvoked {
            skill: "forge".to_string(),
            session_id: Some("sess-123".to_string()),
        };
        let cytokine = event_to_cytokine(event);
        assert!(matches!(cytokine.family, CytokineFamily::Il2));
        assert!(matches!(cytokine.severity, ThreatLevel::Low));
    }

    // ── Dual-emit bridge tests ─────────────────────────────────

    #[test]
    fn test_cytokine_to_signal_tool_blocked() {
        let event = HookEvent::ToolBlocked {
            tool: "Write".to_string(),
            hook: "unwrap-guardian".to_string(),
            reason: "unwrap() detected".to_string(),
        };
        let cytokine = event_to_cytokine(event);
        let signal = cytokine_to_signal(&cytokine);

        assert!(
            signal.signal_type.starts_with("cytokine:tnf_alpha:"),
            "Expected tnf_alpha prefix, got: {}",
            signal.signal_type
        );
        assert!(
            signal.signal_type.contains("hook_blocked:unwrap-guardian"),
            "Expected hook name in signal_type, got: {}",
            signal.signal_type
        );
        assert_eq!(signal.priority, Priority::High);
        assert_eq!(
            signal.data.get("family").map(String::as_str),
            Some("tnf_alpha")
        );
        assert_eq!(
            signal.data.get("payload_tool").map(String::as_str),
            Some("Write")
        );
    }

    #[test]
    fn test_cytokine_to_signal_check_failed() {
        let event = HookEvent::CheckFailed {
            hook: "compile-verifier".to_string(),
            error: "cargo check failed".to_string(),
        };
        let cytokine = event_to_cytokine(event);
        let signal = cytokine_to_signal(&cytokine);

        assert!(signal.signal_type.starts_with("cytokine:il6:"));
        assert_eq!(signal.priority, Priority::High);
        assert_eq!(
            signal.data.get("severity").map(String::as_str),
            Some("high")
        );
    }

    #[test]
    fn test_cytokine_to_signal_hook_completed() {
        let event = HookEvent::HookCompleted {
            hook: "panic-detector".to_string(),
            duration_us: 450,
            verdict: "pass".to_string(),
        };
        let cytokine = event_to_cytokine(event);
        let signal = cytokine_to_signal(&cytokine);

        assert!(signal.signal_type.starts_with("cytokine:tgf_beta:"));
        assert_eq!(signal.priority, Priority::Telemetry);
        assert_eq!(
            signal.data.get("scope").map(String::as_str),
            Some("autocrine")
        );
    }

    #[test]
    fn test_severity_to_priority_mapping() {
        assert_eq!(
            severity_to_priority(&ThreatLevel::Critical),
            Priority::Critical
        );
        assert_eq!(severity_to_priority(&ThreatLevel::High), Priority::High);
        assert_eq!(severity_to_priority(&ThreatLevel::Medium), Priority::Normal);
        assert_eq!(severity_to_priority(&ThreatLevel::Low), Priority::Normal);
        assert_eq!(
            severity_to_priority(&ThreatLevel::Trace),
            Priority::Telemetry
        );
    }
}
